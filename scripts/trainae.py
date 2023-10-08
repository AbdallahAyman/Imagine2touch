# standard libraries
import argparse
import shutil
import sys
from reskin.data_collection.utils import search_folder
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from pickle import dump, load
import hydra
from omegaconf import OmegaConf

# from ax.service.managed_loop import optimize
import matplotlib.pyplot as plt

# relative modules
from reskin.models.utils import (
    preprocess_object_data,
    prepocess_task_data,
    infer,
    train_touch_to_image,
    reorder_shuffled,
)
from reskin.models.models import AE
from reskin.reskin_calibration import dataset
from reskin.utils.utils import NotAdaptedError


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="10", help="model id")
    # add parser argument for infer output type
    parser.add_argument(
        "--infer_type", type=str, default="image", help="infer output type"
    )
    # add parser argument for model.encode_image
    parser.add_argument("--encode_image", default="true", help="encode image")
    parser.add_argument("--encode_tactile", default="false", help="encode tactile")
    parser.add_argument(
        "--overwrite_model", action="store_true", help="overwrite model"
    )
    args = parser.parse_args()
    return args


def parse_configs():
    cfg_masks = cfg_masks.masks
    cfg.image_size = [int(cfg.image_size), int(cfg.image_size)]
    test_objects_array = cfg.test_objects_names.split(",")
    test_objects_string = ""
    for i, object in enumerate(test_objects_array):
        if i < len(test_objects_array) - 1:
            test_objects_string += object + "_"
        else:
            test_objects_string += object
    return cfg, cfg_masks, test_objects_string


def init_config():
    OmegaConf.register_new_resolver("model_id", lambda x: args.model_id)
    OmegaConf.register_new_resolver("infer_type", lambda x: args.infer_type)
    OmegaConf.register_new_resolver(
        "encode_tactile",
        lambda x: True if args.encode_tactile.lower() == "true" else False,
    )
    OmegaConf.register_new_resolver(
        "encode_image", lambda x: True if args.encode_image.lower() == "true" else False
    )
    OmegaConf.register_new_resolver(
        "image_factor", lambda x, y: 1 if y else 3 if x == "rgb" else 1
    )
    OmegaConf.register_new_resolver("rgb_gray_factor", lambda x: 1 if x else 3)
    # load script configurations
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("trainae.yaml")
    cfg_masks = hydra.compose("generate_masks.yaml")
    return cfg, cfg_masks


if __name__ == "__main__":
    args = parse_arguments()
    cfg, cfg_masks = init_config()
    repo_path = search_folder("/", cfg.repository_dir)
    cfg, cfg_masks, test_objects_string = parse_configs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data preprocessing and loading for training
    if not cfg.inference:
        # get data and scaling parameters
        (
            scaler_images,
            scaler_rgb,
            scaler_reskin,
            tactile_targets_train,
            target_images_train,
            tactile_input_train,
            rgb_images_train,
            _,
            mean_images,
            std_images,
            mean_reskin,
            std_reskin,
            target_masks_train,
        ) = preprocess_object_data(cfg, repo_path, cfg.objects_names)
        # create training loader
        TrainSet = dataset.Touch2imageSet(
            tactile_input_train,
            tactile_targets_train,
            target_images_train,
            rgb_images_train,
            target_masks_train,
        )
        train_loader = DataLoader(TrainSet, batch_size=cfg.parameters.bs, shuffle=True)
    # model and scaling parameters loading for inference and validation
    else:
        # load scaling parameters
        mean_reskin = np.load(
            f"{cfg.model_path}/{cfg.model_id}/reskin_scaling_mean.npy",
            allow_pickle=True,
        )
        std_reskin = np.load(
            f"{cfg.model_path}/{cfg.model_id}/reskin_scaling_std.npy", allow_pickle=True
        )
        mean_images = np.load(
            f"{cfg.model_path}/{cfg.model_id}/images_scaling_mean.npy",
            allow_pickle=True,
        )
        std_images = np.load(
            f"{cfg.model_path}/{cfg.model_id}/images_scaling_std.npy", allow_pickle=True
        )
        scaler_reskin = load(
            open(f"{cfg.model_path}/{cfg.model_id}/reskin_scaling_quantile.pkl", "rb")
        )
        scaler_images = load(
            open(f"{cfg.model_path}/{cfg.model_id}/images_scaling_quantile.pkl", "rb")
        )
        scaler_rgb = load(
            open(f"{cfg.model_path}/{cfg.model_id}/rgb_scaling_quantile.pkl", "rb")
        )
        # initialize and load model with trainae.yaml parameters, only __target__  is required from model.yaml
        model_instance = hydra.compose("model.yaml")
        model_instance = {
            key: cfg.model[key] if key in cfg.model else model_instance[key]
            for key in model_instance
        }
        model = hydra.utils.instantiate(model_instance).to(device)
        model.load_state_dict(torch.load(f"{cfg.model_path}/{cfg.model_id}/ae_model"))
    # data preprocessing and loading for inference and validation
    if not cfg.offline_task:
        # get test RobustRescalerand validation data with the training scaling parameters
        (
            tactile_targets_test,
            image_targets_test,
            tactile_input_test,
            rgb_images_test,
            target_masks_test,
            preprocessing_indeces_list_test,
            tactile_targets_validation,
            image_targets_validation,
            tactile_input_validation,
            rgb_images_validation,
            target_masks_validation,
            preprocessing_indeces_list_validation,
        ) = preprocess_object_data(
            cfg,
            cfg.test_objects_names,
            mean_images,
            std_images,
            mean_reskin,
            std_reskin,
            scaler_images,
            scaler_rgb,
            scaler_reskin,
            train=False,
        )
        # remove a potential extra dimension from the preprocessing indeces list
        preprocessing_indeces_list_test = np.squeeze(preprocessing_indeces_list_test)
        # create test and validation sets
        TestSet = dataset.Touch2imageSet(
            tactile_input_test,
            tactile_targets_test,
            image_targets_test,
            rgb_images_test,
            target_masks_test,
            task=cfg.offline_task,
        )
        ValidationSet = dataset.Touch2imageSet(
            tactile_input_validation,
            tactile_targets_validation,
            image_targets_validation,
            rgb_images_validation,
            target_masks_validation,
            task=cfg.offline_task,
        )
    else:
        raise NotAdaptedError(
            "offline task testing is not adapted for current version of code"
        )
        (
            tactile_targets_test,
            tactile_input_test,
            preprocessing_indeces_list,
        ) = prepocess_task_data(
            cfg,
            cfg.test_objects_names,
            mean_reskin=mean_reskin,
            std_reskin=std_reskin,
            scaler_reskin=scaler_reskin,
        )
        TestSet = dataset.Touch2imageSet(
            tactile_input_test, tactile_targets_test, task=cfg.offline_task
        )
    # create test and validation loaders
    test_loader = DataLoader(TestSet, batch_size=cfg.parameters.bs, shuffle=True)
    if not cfg.inference:
        validation_loader = DataLoader(
            ValidationSet, batch_size=cfg.parameters.bs, shuffle=True
        )

    # train, validate and save the model
    mean_mse_tactile_graph = np.zeros(cfg.model.tactile_input_shape - 1)
    # create model folder
    model_folder = f"{cfg.out}/{cfg.model_id}"
    if not os.path.exists(model_folder) and not cfg.inference:
        os.makedirs(model_folder)
    elif os.path.exists(model_folder) and not cfg.inference:
        # prompt user to overwrite model folder
        user_input: str
        if not args.overwrite_model:
            user_input = (
                input(
                    f"The folder '{model_folder}' already exists. Do you want to overwrite it? (y/n): "
                )
                .strip()
                .lower()
            )
        else:
            user_input = "y"
        if user_input == "y":
            # If the user confirms by typing 'y', delete the existing folder and recreate it.
            shutil.rmtree(model_folder)
            os.makedirs(model_folder)
        else:
            print("Folder not overwritten")
            # prompt user to exit
            user_input = input("Do you want to exit? (y/n): ").strip().lower()
            if user_input == "y":
                sys.exit(-1)
            else:
                pass
    if not cfg.inference:
        # create a model from `AE` autoencoder class
        if cfg.ablate_embedding:
            ablation = cfg.model.tactile_input_shape
        else:
            ablation = 1 + 1
        for n in range(1, ablation):
            if ablation != cfg.model.tactile_input_shape:
                pass
            else:
                cfg.model.image_embedding_dim = n
            model = AE(**cfg.model).to(device)
            print(model)
            (
                model,
                batch_tactile_features,
                batch_target_images,
                batch_rgb_images,
                batch_target_masks,
                images_output,
                tactile_output,
                rgb_output,
                masks_output,
                mean_mse_tactile_graph[n - 1],
            ) = train_touch_to_image(
                model,
                scaler_images,
                mean_images,
                std_images,
                scaler_rgb,
                train_loader,
                validation_loader,
                cfg,
                cfg_masks,
                device,
                model_folder=model_folder,
            )

            # AutoML #TODO: make this usable
            # def train_evaluate(parameters=None):
            #     model = AE(**cfg.model).to(device)
            #     model,_,_,_,_,_,_,_,_=train_touch_to_image(model,train_loader,cfg,cfg_masks,device,parameters,autoML=False)
            #     torch.save(model.state_dict(), f'{model_folder}/ae_model')
            #     return {'mse': evaluate (model,test_loader,device,infer,cfg)}

            # best_parameters, values, experiment, model = optimize(
            # parameters=[
            #     {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            # ],
            # evaluation_function=train_evaluate,
            # objective_name='mse',
            # )

            # save scaling parameters
            with open(
                f"{model_folder}/reskin_scaling_mean.npy", "wb"
            ) as reskin_scaling_mean_file:
                np.save(reskin_scaling_mean_file, mean_reskin)
            with open(
                f"{model_folder}/reskin_scaling_std.npy", "wb"
            ) as reskin_scaling_std_file:
                np.save(reskin_scaling_std_file, std_reskin)
            with open(
                f"{model_folder}/images_scaling_mean.npy", "wb"
            ) as images_scaling_mean_file:
                np.save(images_scaling_mean_file, mean_images)
            with open(
                f"{model_folder}/images_scaling_std.npy", "wb"
            ) as images_scaling_std_file:
                np.save(images_scaling_std_file, std_images)
            with open(
                f"{model_folder}/reskin_scaling_quantile.pkl", "wb"
            ) as reskin_scaling_quantile_file:
                dump(scaler_reskin, reskin_scaling_quantile_file)
            with open(
                f"{model_folder}/images_scaling_quantile.pkl", "wb"
            ) as images_scaling_quantile_file:
                dump(scaler_images, images_scaling_quantile_file)
            with open(
                f"{model_folder}/rgb_scaling_quantile.pkl", "wb"
            ) as rgb_scaling_quantile_file:
                dump(scaler_rgb, rgb_scaling_quantile_file)
            # save the model
            torch.save(model.state_dict(), f"{model_folder}/ae_model")
    # save mean_mse_tactile_graph
    if cfg.ablate_embedding:
        np.save(f"{model_folder}/mean_mse_tactile_graph.npy", mean_mse_tactile_graph)
        # # plot the mean mse graph
        plt.plot(np.arange(1, 15), mean_mse_tactile_graph)
        plt.xlabel("embedding dimension")
        plt.ylabel("mean mse")
        plt.savefig(f"{model_folder}/mean_mse_tactile_graph.png")
        plt.show()

    # # inference
    # # empty lists to fill with test data
    original_target_images = None
    original_rgb_images = None
    depth_reconstructions = []
    rgb_reconstructions = []
    loader_indeces_list = []
    # inference loop
    with torch.no_grad():
        (
            original_tactile,
            original_rgb_images,
            original_target_images,
            original_target_masks,
            tactile_reconstructions,
            depth_reconstructions,
            rgb_reconstructions,
            masks_reconstructions,
            loader_indeces_list,
        ) = infer(model, cfg, device, test_loader)
        # post processing; reorder the test data
        loader_indeces_list = np.squeeze(
            torch.cat(loader_indeces_list, dim=0).cpu().numpy().reshape(1, -1)
        )
        masks_reconstructions = reorder_shuffled(
            masks_reconstructions, loader_indeces_list, preprocessing_indeces_list_test
        )
        masks_reconstructions = torch.where(
            masks_reconstructions > cfg.masks_threshold, 1.0, 0
        )
        depth_reconstructions = reorder_shuffled(
            depth_reconstructions, loader_indeces_list, preprocessing_indeces_list_test
        )
        tactile_reconstructions = reorder_shuffled(
            tactile_reconstructions,
            loader_indeces_list,
            preprocessing_indeces_list_test,
        )
        if rgb_reconstructions is not None:
            rgb_reconstructions = reorder_shuffled(
                rgb_reconstructions,
                loader_indeces_list,
                preprocessing_indeces_list_test,
            )
        original_target_masks = reorder_shuffled(
            original_target_masks, loader_indeces_list, preprocessing_indeces_list_test
        )
        original_target_images = reorder_shuffled(
            original_target_images, loader_indeces_list, preprocessing_indeces_list_test
        )
        original_rgb_images = reorder_shuffled(
            original_rgb_images, loader_indeces_list, preprocessing_indeces_list_test
        )
        original_tactile = reorder_shuffled(
            original_tactile, loader_indeces_list, preprocessing_indeces_list_test
        )
    # log predictions
    if cfg.save:
        depth_path = f"{model_folder}/images_reconstructed/depth/"
        masks_path = f"{model_folder}/images_reconstructed/masks/"
        tactile_path = f"{model_folder}/tactile_reconstructed/"
        os.makedirs(os.path.dirname(tactile_path), exist_ok=True)
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        os.makedirs(os.path.dirname(masks_path), exist_ok=True)
        # save_images(cfg, masks_reconstructions, masks_path, type="masks")
        # save_images(
        #     cfg,
        #     depth_reconstructions,
        #     depth_path,
        #     scaler_images,
        #     mean_images,
        #     std_images,
        #     "depth",
        # )
        save_tactile(
            tactile_reconstructions,
            scaler_reskin,
            mean_reskin,
            std_reskin,
            tactile_path,
        )
    # plot predictions
    # plot_touch_to_image(
    #     cfg,
    #     original_tactile,
    #     original_target_images,
    #     original_rgb_images,
    #     original_target_masks,
    #     depth_reconstructions,
    #     rgb_reconstructions,
    #     tactile_reconstructions,
    #     masks_reconstructions,
    #     scaler_images,
    #     std_images,
    #     mean_images,
    #     scaler_rgb,
    #     show=cfg.show_samples,
    # )

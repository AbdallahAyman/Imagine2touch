import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from reskin.utils.utils import NotAdaptedError


class simpleMLP(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        hidden_dims=[64, 64],
        activation_fn=nn.Tanh,
        output_activation=None,
    ):
        super(simpleMLP, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        layer_dims = [n_input] + hidden_dims + [n_output]
        layers = []

        for d in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[d], layer_dims[d + 1]))
            if d < len(layer_dims) - 2:
                layers.append(activation_fn())

        if output_activation is not None:
            layers.append(output_activation())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class vanilla_model(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        feature_dim=20,
        feat_hidden=[64, 64],
        activation_fn=nn.Tanh,
        feat_activation=None,
        output_hidden=[64, 64],
        output_activation=None,
        pred_Fz=True,
        pred_Fxy=False,
    ):
        super(vanilla_model, self).__init__()
        self.n_input = n_input
        # self.n_output = 2 + int(pred_Fz) + 2*int(pred_Fxy)
        self.n_output = n_output
        self.feature_dim = feature_dim
        self.feat_model = simpleMLP(
            n_input=n_input,
            n_output=feature_dim,
            hidden_dims=feat_hidden,
            activation_fn=activation_fn,
            output_activation=feat_activation,
        )
        self.output_model = simpleMLP(
            feature_dim,
            self.n_output,
            hidden_dims=output_hidden,
            activation_fn=activation_fn,
            output_activation=output_activation,
        )

    def forward(self, sens):
        return self.output_model(self.get_feature(sens))

    def get_feature(self, sens):
        return self.feat_model(sens)

    def get_out_from_feature(self, feature):
        return self.output_model(feature)


# credit to https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
class AE(nn.Module):
    def __init__(
        self,
        tactile_encoder_hidden,
        tactile_decoder_hidden,
        images_decoder_hidden,
        rgb_images_decoder_hidden=-1,
        tactile_embedding_dim=5,
        tactile_input_shape=15,
        rgbd=False,
        rgb_gray=True,
        aux_reconstruction=False,
        cnn_images_encoder=True,
        images_encoder_hidden=0,
        images_input_shape=0,
        images_output_shape=0,
        image_embedding_dim=0,
        connect_image_encoder_in_encode_image=False,
        encode_image=False,
        reuse_tactile_decoder_in_encode_image=False,
        var_ae=False,
        fuse_encode_image_code=False,
        encode_tactile=True,
        connect_tactile_encoder_in_encode_image=False,
    ):
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.tactile_encoder_hidden = tactile_encoder_hidden
        self.images_encoder_hidden = images_encoder_hidden
        self.tactile_decoder_hidden = tactile_decoder_hidden
        self.images_decoder_hidden = images_decoder_hidden
        self.rgb_images_decoder_hidden = rgb_images_decoder_hidden
        self.images_input_shape = images_input_shape
        self.images_output_shape = images_output_shape
        self.rgbd = rgbd
        self.rgb_gray = rgb_gray
        self.aux_reconstruction = aux_reconstruction
        self.cnn_images_encoder = cnn_images_encoder
        self.connect_image_encoder_in_encode_image = (
            connect_image_encoder_in_encode_image
        )
        self.connect_tactile_encoder_in_encode_image = (
            connect_tactile_encoder_in_encode_image
        )
        self.encode_image = encode_image
        self.reuse_tactile_decoder_in_encode_image = (
            reuse_tactile_decoder_in_encode_image
        )
        self.var_ae = var_ae
        self.fuse_encode_image_code = fuse_encode_image_code
        self.encode_tactile = encode_tactile
        if not encode_tactile:
            tactile_embedding_dim = image_embedding_dim
        if not rgb_gray:
            image_factor = 3
        else:
            image_factor = 1
        super().__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(8)

        # Max Pooling Layer
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Embedding Layer
        # self.embedding = nn.Linear(24 * 24 * 16, image_embedding_dim)
        self.embedding = nn.Linear(32, image_embedding_dim)
        if self.encode_tactile:
            self.encoder_input_layer = nn.Linear(
                in_features=tactile_input_shape,
                out_features=self.tactile_encoder_hidden,
            )
            self.encoder_hidden_layer = nn.Linear(
                in_features=self.tactile_encoder_hidden,
                out_features=self.tactile_encoder_hidden,
            )
            self.encoder_output_layer = nn.Linear(
                in_features=self.tactile_encoder_hidden,
                out_features=tactile_embedding_dim,
            )
            self.encoder_output_layer_var = nn.Linear(
                in_features=self.tactile_encoder_hidden,
                out_features=tactile_embedding_dim,
            )
        self.decoder_hidden_layer_tactile = nn.Linear(
            in_features=tactile_embedding_dim,
            out_features=self.tactile_decoder_hidden,
        )
        self.decoder_output_layer_tactile = nn.Linear(
            in_features=self.tactile_decoder_hidden,
            # in_features=tactile_embedding_dim,
            out_features=tactile_input_shape,
        )
        self.decoder_output_layer_mask = nn.Linear(
            in_features=self.images_decoder_hidden,
            # in_features=image_embedding_dim,
            out_features=self.images_output_shape,
        )
        if self.images_encoder_hidden > -1 and not self.cnn_images_encoder:
            self.encoder_image_input_layer = nn.Linear(
                in_features=images_input_shape,
                out_features=self.images_encoder_hidden,
            )
            self.encoder_image_hidden_layer = nn.Linear(
                in_features=self.images_encoder_hidden,
                out_features=self.images_encoder_hidden,
            )
            self.encoder_image_output_layer = nn.Linear(
                in_features=self.images_encoder_hidden,
                out_features=image_embedding_dim,
            )
            self.encoder_image_output_layer_var = nn.Linear(
                in_features=self.images_encoder_hidden,
                out_features=image_embedding_dim,
            )
        elif self.images_encoder_hidden <= -1 and self.cnn_images_encoder:
            self.encoder_image_output_layer_var = nn.Linear(
                in_features=32,
                out_features=image_embedding_dim,
            )
        if image_embedding_dim == -1:
            image_embedding_dim = tactile_embedding_dim
        self.decoder_hidden_layer_image = nn.Linear(
            in_features=image_embedding_dim, out_features=self.images_decoder_hidden
        )
        self.decoder_output_layer_image = nn.Linear(
            in_features=self.images_decoder_hidden, out_features=images_input_shape
        )
        # self.decoder_hidden_layer_rgb = nn.Linear(
        #     in_features=tactile_embedding_dim, out_features=self.rgb_images_decoder_hidden)
        # self.decoder_output_layer_rgb = nn.Linear(
        #     in_features=self.rgb_images_decoder_hidden, out_features=images_input_shape * image_factor)
        self.flatten = nn.Flatten()
        if self.encode_tactile:
            self.encoder_input_layer.apply(init_weights)
            self.encoder_hidden_layer.apply(init_weights)
            self.encoder_output_layer.apply(init_weights)
        self.decoder_hidden_layer_tactile.apply(init_weights)
        self.decoder_output_layer_tactile.apply(init_weights)
        self.decoder_hidden_layer_image.apply(init_weights)
        self.decoder_output_layer_image.apply(init_weights)
        self.decoder_output_layer_mask.apply(init_weights)
        # The next condition is redundant to check for twice, it is here for clarity and organisation.
        if self.images_encoder_hidden > -1:
            self.encoder_image_input_layer.apply(init_weights)
            self.encoder_image_hidden_layer.apply(init_weights)
            self.encoder_image_output_layer.apply(init_weights)
        if self.rgbd:
            self.decoder_hidden_layer_rgb.apply(init_weights)
            self.decoder_output_layer_rgb.apply(init_weights)

        self.embedding.apply(init_weights)
        self.pool1.apply(init_weights)
        self.pool2.apply(init_weights)
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        self.conv3.apply(init_weights)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_2(self, tactile_data, image_data):
        x = nn.functional.tanh(self.batchnorm1(self.conv1(image_data)))
        x = self.pool1(x)
        x = nn.functional.tanh(self.batchnorm2(self.conv2(x)))
        x = self.pool2(x)
        x = nn.functional.tanh(self.batchnorm3(self.conv3(x)))
        # x = self.pool3(x)
        x = self.flatten(x)
        # Embedding Layer
        x = self.embedding(x)
        x = torch.sigmoid(x)
        # x = self.decoder_hidden_layer_tactile(x)
        # x = torch.relu(x)

        # def forward(self, tactile_data, image_data):
        #     x = nn.functional.relu(self.conv1(image_data))
        #     x = self.pool1(x)
        #     x = nn.functional.relu(self.conv2(x))
        #     x = self.pool2(x)
        #     x = nn.functional.relu(self.conv3(x))
        #     # x = self.pool3(x)
        #     x = self.flatten(x)
        #     # Embedding Layer
        #     x = self.embedding(x)
        #     x = torch.sigmoid(x)
        #     # x = self.decoder_hidden_layer_tactile(x)
        #     # x = torch.relu(x)

        tactile = self.decoder_output_layer_tactile(x)
        mask = self.decoder_output_layer_mask(x)
        # Fake weird outputs
        depth = torch.zeros_like(image_data)
        # mask = torch.zeros_like(image_data)
        code_tactile = torch.zeros((0))
        code_image = torch.zeros((0))
        return tactile, depth, mask, code_tactile, code_image

    def forward(self, tactile_data, image_data):
        # encode tactile data
        if self.encode_tactile:
            activation_tactile = self.encoder_input_layer(tactile_data)
            activation_tactile = torch.relu(activation_tactile)
            hidden_layer_tactile = self.encoder_hidden_layer(activation_tactile)
            hidden_layer_tactile = torch.relu(hidden_layer_tactile)
            code_tactile = self.encoder_output_layer(hidden_layer_tactile)
            if self.var_ae:
                code_logvar_tactile = self.encoder_output_layer_var(
                    hidden_layer_tactile
                )
                code_tactile = self.reparameterize(code_tactile, code_logvar_tactile)
                # code_tactile = torch.sigmoid(code_tactile)
        if self.cnn_images_encoder:
            # x = nn.functional.relu(self.conv1(image_data))
            # x = self.pool1(x)
            # x = nn.functional.relu(self.conv2(x))
            # x = self.pool2(x)
            # x = nn.functional.relu(self.conv3(x))
            # # x = self.pool3(x)
            x = nn.functional.relu(self.batchnorm1(self.conv1(image_data)))
            x = self.pool1(x)
            x = nn.functional.relu(self.batchnorm2(self.conv2(x)))
            x = self.pool2(x)
            x = nn.functional.relu(self.batchnorm3(self.conv3(x)))
            x = torch.sigmoid(self.flatten(x))
            # Embedding Layer
            code_image = self.embedding(x)

        ## decode image data using one embedding architecture
        if self.encode_image:
            # encode image data
            if not self.cnn_images_encoder:
                activation_image = self.encoder_image_input_layer(image_data)
                activation_image = torch.relu(activation_image)
                hidden_layer_image = self.encoder_image_hidden_layer(activation_image)
                hidden_layer_image = torch.relu(hidden_layer_image)
                code_image = self.encoder_image_output_layer(hidden_layer_image)
                if self.var_ae:
                    code_logvar_image = self.encoder_image_output_layer_var(
                        hidden_layer_image
                    )
                    code_image = self.reparameterize(code_image, code_logvar_image)
                    # code_image = torch.sigmoid(code_image)
            elif self.var_ae:
                code_logvar_image = self.encoder_image_output_layer_var(x)
                code_image = self.reparameterize(code_image, code_logvar_image)
                # code_image = torch.sigmoid(code_image)

            # decode image data
            if self.reuse_tactile_decoder_in_encode_image:
                decoder_hidden_layer_image = self.decoder_hidden_layer_tactile
            else:
                decoder_hidden_layer_image = self.decoder_hidden_layer_image
            if self.connect_image_encoder_in_encode_image and self.encode_tactile:
                if self.fuse_encode_image_code:
                    activation_image = decoder_hidden_layer_image(code_tactile)
                else:
                    activation_image = decoder_hidden_layer_image(code_image)
                activation_image = torch.relu(activation_image)
                activation_image = self.decoder_output_layer_image(activation_image)
            elif not self.connect_image_encoder_in_encode_image and self.encode_tactile:
                activation_image = decoder_hidden_layer_image(code_tactile)
                activation_image = torch.relu(activation_image)
                activation_image = self.decoder_output_layer_image(activation_image)
            else:
                activation_image = decoder_hidden_layer_image(code_image)
                activation_image = torch.relu(activation_image)
                activation_image = self.decoder_output_layer_image(activation_image)
            mask = torch.sigmoid(activation_image)
            depth = torch.relu(activation_image)
        else:
            if not self.encode_tactile:
                raise Exception("encode either image or tactile data")
            code_image = code_tactile
            activation_image = self.decoder_hidden_layer_image(code_tactile)
            activation_image = torch.relu(activation_image)
            activation_mask = self.decoder_output_layer_mask(activation_image)
            activation_image = self.decoder_output_layer_image(activation_image)
            depth = torch.relu(activation_image)
            mask = torch.sigmoid(activation_mask)

        if (
            self.aux_reconstruction
        ):  # All predictions other than the main image prediction are considered auxiliary predictions.
            if (
                self.encode_tactile
                and self.encode_image
                and self.connect_tactile_encoder_in_encode_image
            ):
                activation_tactile = self.decoder_hidden_layer_tactile(code_tactile)
            elif self.encode_tactile and not self.encode_image:
                activation_tactile = self.decoder_hidden_layer_tactile(code_tactile)
            elif (
                not self.encode_tactile
                and self.encode_image
                and not self.connect_tactile_encoder_in_encode_image
            ):
                code_tactile = torch.zeros_like(code_image)
                activation_tactile = self.decoder_hidden_layer_tactile(code_image)
            activation_tactile = torch.relu(activation_tactile)
            tactile = self.decoder_output_layer_tactile(activation_tactile)
            return tactile, depth, mask, code_tactile, code_image
        else:
            raise NotAdaptedError(
                "(No Aux reconstruction) is not adapted to the current version of code."
            )
            return depth, mask

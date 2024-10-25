import copy
from typing import Optional

from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import DropPath
import torch
import torch.nn as nn
from dacite import from_dict
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig


#Frontend
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.backbones.conv1d_extractor import Conv1dResNet
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet
from xlstm.xlstm.xlstm_block_stack import xLSTMBlockStack
from xlstm.xlstm.xlstm_lm_model import xLSTMLMModelConfig
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class XlstmEncoder(nn.Module):
    """This class implements the Conmamba encoder.
    """

    def __init__(
        self,
        args,
        d_model,
        input_layer="conv2d",
        relu_type="prelu",
        a_upsample_ratio=1,
    ):
        super().__init__()

        # -- frontend module.
        if input_layer == "conv1d":
            self.frontend = Conv1dResNet(relu_type=relu_type, a_upsample_ratio=a_upsample_ratio)
        elif input_layer == "conv3d":
            self.frontend = Conv3dResNet(relu_type=relu_type)
        else:
            self.frontend = None  
        
        # -- backend module.
        if input_layer in ["conv1d", "conv3d"]:
            self.embed = torch.nn.Linear(512, d_model)
            torch.nn.init.xavier_normal_(self.embed.weight)
        elif input_layer is None:
            self.embed = None

        cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(args.xlstm_cfg), config=DaciteConfig(strict=True))
        self.layers = xLSTMBlockStack(cfg)      
         
        # from speechbrain.nnet.normalization import LayerNorm

        self.norm = LayerNorm(d_model)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor, optional
            The mask for the src sequence.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch.
        pos_embs: torch.Tensor, torch.nn.Module,
            Module or tensor containing the input sequence positional embeddings
            If custom pos_embs are given it needs to have the shape (1, 2*S-1, E)
            where S is the sequence length, and E is the embedding dimension.
        dynchunktrain_config: Optional[DynChunkTrainConfig]
            Dynamic Chunk Training configuration object for streaming,
            specifically involved here to apply Dynamic Chunk Convolution to the
            convolution module.
        """
        output = src

        if isinstance(self.frontend, (Conv1dResNet, Conv3dResNet)):
            output = self.frontend(output)
       
        if self.embed:
            output = self.embed(output)

        output = self.layers(output)

        output = self.norm(output) #if was commented before

        return output, None
    

class ConXlstmEncoder(nn.Module):
    """This class implements the Conmamba encoder.
    """

    def __init__(
        self,
        args,
        d_model=768,
        linear_units=3072,
        num_blocks=12,
        dropout_rate=0.1,
        input_layer="conv2d",
        normalize_before=True,
        concat_after=False,
        macaron_style=False,
        use_cnn_module=False,
        cnn_module_kernel=31,
        relu_type="prelu",
        a_upsample_ratio=1,
        layerscale=False,
        positional_dropout_rate=0.1,
        init_values=0.,
        drop_path=0.,
        pos_enc_class=PositionalEncoding,
    ):
        super().__init__()

        # -- frontend module.
        if input_layer == "conv1d":
            self.frontend = Conv1dResNet(relu_type=relu_type, a_upsample_ratio=a_upsample_ratio)
        elif input_layer == "conv3d":
            self.frontend = Conv3dResNet(relu_type=relu_type)
        else:
            self.frontend = None  
        
        # -- backend module.
        if input_layer in ["conv1d", "conv3d"]:
            self.embed = torch.nn.Linear(512, d_model)
            torch.nn.init.xavier_normal_(self.embed.weight)
        elif input_layer is None:
            self.embed = None
        # if input_layer in ["conv1d", "conv3d"]:
        #     self.embed = torch.nn.Sequential(torch.nn.Linear(512, d_model), pos_enc_class(d_model, positional_dropout_rate))
        # elif input_layer is None:
        #     self.embed = torch.nn.Sequential(pos_enc_class(d_model, positional_dropout_rate))


        self.normalize_before = normalize_before
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (d_model, linear_units, dropout_rate)

        xlstm_config = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(args.xlstm_cfg), config=DaciteConfig(strict=True))
        encoder_xlstm = xLSTMBlockStack(xlstm_config)  


        convolution_layer = ConvolutionModule
        convolution_layer_args = (d_model, cnn_module_kernel)

        self.encoders = repeat(
            num_blocks,
            lambda: ConXlstmEncoderLayer(
                d_model,
                encoder_xlstm,
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                macaron_style,
                layerscale,
                init_values,
                drop_path,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(d_model)

    def forward(self, xs, masks):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.frontend, (Conv1dResNet, Conv3dResNet)):
            xs = self.frontend(xs)

        xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks
    

class ConXlstmEncoderLayer(nn.Module):
    
    def __init__(
        self,
        size,
        encoder_xlstm,
        feed_forward,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        macaron_style=False,
        layerscale=False,
        init_values=0.,
        drop_path=0.,
    ):
        """Construct an EncoderLayer object."""
        super(ConXlstmEncoderLayer, self).__init__()
        self.encoder_xlstm = encoder_xlstm
        self.feed_forward = feed_forward
        self.ff_scale = 1.0
        self.conv_module = conv_module
        self.macaron_style = macaron_style
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_xlstm = LayerNorm(size)  # for the XLSTM module
        if self.macaron_style:
            self.feed_forward_macaron = copy.deepcopy(feed_forward)
            self.ff_scale = 0.5
            # for another FNN module in macaron style
            self.norm_ff_macaron = LayerNorm(size)
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.layerscale = layerscale
        if layerscale:
            self.gamma_ff = nn.Parameter(init_values * torch.ones((size,)), requires_grad=True)
            self.gamma_mha = nn.Parameter(init_values * torch.ones((size,)), requires_grad=True)
            if self.macaron_style:
                self.gamma_ff_macaron = nn.Parameter(init_values * torch.ones((size,)), requires_grad=True)
            if self.conv_module is not None:
                self.gamma_conv = nn.Parameter(init_values * torch.ones((size,)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        :param torch.Tensor x_input: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # whether to use macaron style
        if self.macaron_style:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            if self.layerscale:
                x = residual + self.drop_path(
                    self.ff_scale * self.dropout(self.gamma_ff_macaron * self.feed_forward_macaron(x))
                )
            else:
                x = residual + self.drop_path(self.ff_scale * self.dropout(self.feed_forward_macaron(x)))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)
            

        # xlstm module
        residual = x
        if self.normalize_before:
            x = self.norm_xlstm(x)

        x_xlstm = self.encoder_xlstm(x)
        
        if self.concat_after:
            x_concat = torch.cat((x, x_xlstm), dim=-1)
            if self.layerscale:
                x = residual + self.drop_path(self.gamma_mha * self.concat_linear(x_concat))
            else:
                x = residual + self.drop_path(self.concat_linear(x_concat))
        else:
            if self.layerscale:
                x = residual + self.drop_path(self.dropout(self.gamma_mha * x_xlstm))
            else:
                x = residual + self.drop_path(self.dropout(x_xlstm))
        if not self.normalize_before:
            x = self.norm_xlstm(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            if self.layerscale:
                x = residual + self.drop_path(self.dropout(self.gamma_conv * self.conv_module(x)))
            else:
                x = residual + self.drop_path(self.dropout(self.conv_module(x)))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        if self.layerscale:
            x = residual + self.drop_path(self.ff_scale * self.dropout(self.gamma_ff * self.feed_forward(x)))
        else:
            x = residual + self.drop_path(self.ff_scale * self.dropout(self.feed_forward(x)))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask
import abc

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer
from models.train_utils import *


# noinspection PyAbstractClass
class MetaLearner(GraphTrainer):

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams, label_names, batch_size):
        """
        Inputs
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            n_inner_updates - Number of inner loop updates to perform
        """
        super().__init__(validation_sets=['val'])
        self.save_hyperparameters()

        self.n_inner_updates = model_params['n_inner_updates']
        self.n_inner_updates_test = model_params['n_inner_updates_test']

        self.lr_output = self.hparams.optimizer_hparams['lr_output']
        self.lr_inner = self.hparams.optimizer_hparams['lr_inner']

        # flipping the weights
        pos_weight = 1 // model_params["class_weight"]['train'][1]
        print(f"Using positive weight: {pos_weight}")
        self.loss_module = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model = GatNet(model_params)

        self.automatic_optimization = False

    def configure_optimizers(self):
        opt_params = self.hparams.optimizer_hparams
        # scheduler = MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
        optimizer, scheduler = self.get_optimizer(opt_params['lr'], opt_params['lr_decay_epochs'])
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode="train")

        # Returning None means skipping the default training optimizer steps by PyTorch Lightning
        return None

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="val")
        torch.set_grad_enabled(False)

    @abc.abstractmethod
    def outer_loop(self, batch, mode):
        raise NotImplementedError

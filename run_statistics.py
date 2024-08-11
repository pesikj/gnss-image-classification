from dataclasses import dataclass


@dataclass
class RunStatistics:
    cross_validation_iteration: int
    model: str
    epoch: int
    epoch_total: int
    phase: str
    epoch_loss: float
    epoch_acc: float

    def get_conf_matrix_filename(self):
        return f'{self.model}_{self.cross_validation_iteration}.xlsx'

from data.dvr import DVRDataset
from data.dvr_eval import DVREvalDataset
from data.srn import SRNDataset
from data.srn_eval import SRNEvalDataset

dataset_dict = {
    'srn': SRNDataset,
    'dvr': DVRDataset,
}

eval_dataset_dict = {
    'srn': SRNEvalDataset,
    'dvr': DVREvalDataset,
}

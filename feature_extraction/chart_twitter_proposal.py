# from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse


class ChartTwitterDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_path_list = list(tqdm(image_dir.iterdir()))
        self.n_images = len(self.image_path_list)

        # self.transform = image_transform

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_id = image_path.stem

        img = cv2.imread(str(image_path))

        return {
            'img_id': image_id,
            'img': img
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--dataroot', type=str,
                        default='/dvmm-filer2/projects/mingyang/semafor/Twitter-COMMS')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'val', 'test'])
    parser.add_argument('--proposal_type', type=str, default='object_detector', choices=['object_detector', 'chart_element'])

    args = parser.parse_args()

    SPLIT2DIR = {
        'train': 'train',
        'val': 'val',
        'test': 'test',
    }
    

    if args.proposal_type == "object_detector":
        from detectron2_proposal_maxnms import collate_fn, NUM_OBJECTS, DIM
    else:
        from detectron_chart_feature_extractor import collate_fn, NUM_OBJECTS, DIM

    chartsum_dir = Path(args.dataroot).resolve()
    chartsum_img_dir = chartsum_dir.joinpath(SPLIT2DIR[args.split]).joinpath("images")

    dataset_name = 'ChartTwitterCOMMS'

    out_dir = chartsum_dir.joinpath('features')
    if not out_dir.exists():
        out_dir.mkdir()

    print('Load images from', chartsum_img_dir)
    print('# Images:', len(list(chartsum_img_dir.iterdir())))

    dataset = ChartTwitterDataset(chartsum_img_dir)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    # output_fname = out_dir.joinpath(f'{args.split}_boxes{NUM_OBJECTS}.h5')
    # print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{args.split}_{(NUM_OBJECTS, DIM)}'

    # extract(output_fname, dataloader, desc)

    if args.proposal_type == "object_detector":
        output_fname = out_dir.joinpath(f'{args.split}_boxes{NUM_OBJECTS}.h5')
        print('features will be saved at', output_fname)
        from detectron2_proposal_maxnms import extract
        extract(output_fname, dataloader, desc)
    elif args.proposal_type == "chart_element":
        output_fname = out_dir.joinpath(f'{args.split}_chart_elements.h5')
        from detectron_chart_feature_extractor import extract
        extract(output_fname, dataloader, desc)
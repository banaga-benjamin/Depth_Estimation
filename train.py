from time import time
from os import cpu_count

import dataset
from networks import depth_encoder
from networks import depth_decoder
from networks import depth_convgru

import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def train_step(dataloader: DataLoader, encoder: depth_encoder.DepthEncoder, decoder: depth_decoder.DepthDecoder, convgru: depth_convgru.ConvGru):
    start_time = time( )
    num_batches = len(dataloader.dataset) // dataloader.batch_size
    print("number of batches:", num_batches, "\n")
    for batch, img_batch in enumerate(dataloader):
        for img_seq in img_batch:
            H = 192; W = 640
            target_img = torch.unsqueeze(img_seq[-1], dim = 0)
            encoder_outputs = encoder(target_img)

            for idx in range(len(img_seq) - 1):
                cost_volume = [torch.rand(1, 256, H // 16, W // 16, device = device),
                        torch.rand(1, 128, H // 8, W // 8, device = device),
                        torch.rand(1, 64, H // 4, W // 4, device = device),
                        torch.rand(1, 32, H // 2, W // 2, device = device)]
                convgru(decoder(encoder_outputs, cost_volume))
        torch.cuda.empty_cache( )

        if batch % 100 == 0:
            if batch == 0: continue
            stop_time = time( )
            elapsed_time = (stop_time - start_time)

            print("batches completed:", batch)
            print("time remaining:", (elapsed_time / (batch + 1)) * (num_batches - batch - 1) / 60 , "minutes")
            print("-" * 50, "\n")
    print("\ntime elapsed:", elapsed_time / 60, "minutes")


if __name__ == "__main__":
    print("pytorch version:", torch.__version__)

    device = 'cuda' if torch.cuda.is_available( ) else 'cpu'
    print("using device:", device)

    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(num_magnitude_bins = 7),
        transforms.Resize((192, 640)),
        transforms.ToTensor( )
    ])

    BATCH_SIZE = 6; NUM_WORKERS = cpu_count( )
    train_data = dataset.TrainingData(seq_len = 4, device = device, transform = train_transform)
    train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS // 2,
                                  persistent_workers = True, shuffle = True, drop_last = True)
    
    convgru = depth_convgru.ConvGru(device = device)
    encoder = depth_encoder.DepthEncoder(device = device)
    decoder = depth_decoder.DepthDecoder(device = device)

    EPOCHS = 1
    for epoch in range(EPOCHS):
        with torch.no_grad( ): train_step(train_dataloader, encoder, decoder, convgru)\

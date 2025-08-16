import torch

import torchvision as tv
import torchvision.transforms as transforms

def data(name, data_dir, batch_size, nw=2, size=None, grayscale = False):

    if name == 'mnist':
        h, w = 32, 32
        # Load MNIST and scale up to 32x32, with color channels
        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.Resize((h, w)),
             transforms.ToTensor()])

        dataset = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

        return dataloader, (h, w), len(dataset)

    if name == 'mnist-128':
        h, w = 128, 128
        # Load MNIST and scale up to 32x32, with color channels
        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.Resize((h, w)),
             transforms.ToTensor()])

        dataset = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        return dataloader, (h, w), len(dataset)

    if name == 'ffhq-thumbs':


        h, w = (128, 128) if size is None else size

        # Load MNIST and scale up to 32x32, with color channels
        transform = [] if (h, w) == (128, 128) else [transforms.Resize((h, w))]
        if grayscale:
            transform.append(transforms.Grayscale(num_output_channels=3))
        transform.append(transforms.ToTensor())
        transform = transforms.Compose(transform)

        dataset = (tv.datasets.ImageFolder(root=data_dir, transform=transform))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        return dataloader, (h, w), len(dataset)

    if name == 'ffhq':

        h, w = 1024, 1024
        # Load MNIST and scale up to 32x32, with color channels
        transform = transforms.Compose(
            [transforms.ToTensor()])

        dataset = (tv.datasets.ImageFolder(root=data_dir, transform=transform))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        return dataloader, (h, w), len(dataset)

    fc(name)

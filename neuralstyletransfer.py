import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

# Define the device to be used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the content and style images
content_path = os.path.join(os.getcwd(), "Content_image.jpg")   
style_path = os.path.join(os.getcwd(), "Style_image.jpg")

content_image = Image.open(content_path).convert("RGB")
style_image = Image.open(style_path).convert("RGB")

# Define the transformer to resize and normalize the images
transformer = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform the images
content_tensor = transformer(content_image).unsqueeze(0).to(device)
style_tensor = transformer(style_image).unsqueeze(0).to(device)

# Define the model
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()
        
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input
    
def gram_matrix(input):
    batch_size, channels, h, w = input.size()
    features = input.view(batch_size * channels, h * w)
    G = torch.mm(features, features.t())
    return G.div(batch_size * channels * h * w)

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
# Define the optimizer
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# Initialize the output image
output_img = content_tensor.clone().requires_grad_(True)

# Set the hyperparameters
style_weight = 5e2
content_weight = 1
tv_weight = 0

# Run the style transfer
num_steps = [0, 500, 200, 100, 50]
image_sizes = [256, 512, 1024, 2048, 3620]

for i in range(len(num_steps)):
    optimizer = get_input_optimizer(output_img)
    content_losses = []
    style_losses = []
    tv_losses = []
    iteration = 0
    
    while iteration <= num_steps[i]:
        def closure():
            # Set the gradients to zero
            optimizer.zero_grad()
            # Forward pass through the VGG19 network
            output = cnn(output_img)
        
           # Calculate the content loss
            content_loss = 0
            for cl in content_losses:
                content_loss += cl.loss
            content_loss *= content_weight
        
           # Calculate the style loss
            style_loss = 0
            for sl in style_losses:
                style_loss += sl.loss
            style_loss *= style_weight
        
            # Calculate the total variation loss
            tv_loss = 0
            if tv_weight > 0:
                diff1 = output_img[:, :, 1:, :] - output_img[:, :, :-1, :]
                diff2 = output_img[:, :, :, 1:] - output_img[:, :, :, :-1]
                tv_loss = tv_weight * (torch.sum(torch.abs(diff1)) + torch.sum(torch.abs(diff2)))
        
            # Calculate the total loss
            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
        
            # Print the loss
            if iteration % 50 == 0:
                print("Iteration: {}, Total loss: {:.2f}".format(iteration, total_loss.item()))
            
            iteration += 1
            return total_loss

    # Add content and style losses to the lists
    content_losses.append(ContentLoss(output_img))
    for feature in cnn[:23]:
        if isinstance(feature, nn.Conv2d):
            style_losses.append(StyleLoss(feature(output_img)))
    tv_losses.append(tv_loss)
    
    # Resize the input image to the current size
    new_size = (image_sizes[i], image_sizes[i])
    output_img = nn.functional.interpolate(output_img, size=new_size, mode="bilinear", align_corners=False)
    
    # Run the optimizer
    optimizer.step(closure)
    
    # Clamp the pixel values to the range [0, 1]
    output_img.data.clamp_(0, 1)
    
# Save the output image
output_image = output_img.squeeze(0).cpu().detach()
output_image = transforms.ToPILImage()(output_image)
output_image.save("output_image_{}.jpg".format(image_sizes[i]))


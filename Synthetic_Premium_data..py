import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

Everst_data=pd.read_csv(r'C:\Users\u013709\OneDrive - SCOR\Documents\My Data Sources\Everest_data.csv')
data=Everst_data
data['Attained_age']=data['Age']+data['Duration']
data['Premium_rate']=data['BasePremiumAmount']/data['BaseNARAmount']*10000
exclude_cols = ['Gender','RiskClass']
Coulmns_to_keep=['Attained_age','Premium_rate']
data=data[data['SingleOrJointType']=="Single"]
data=data[data['Gender']=="Male"]
data=data[data['RiskClass']=="NS1"]
data=data[Coulmns_to_keep]
numerical_columns = data.select_dtypes(include=['number']).columns
categorical_columns = data.select_dtypes(include=['object']).columns
#data.iloc[4::5, 1] = np.nan
test_data=data.head(1000)
test_data

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # Use sigmoid to keep output in [0, 1]
        )
    
    def forward(self, x):
        return self.model(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Generate real data
def get_real_data(num_samples):
    ages = np.random.uniform(70, 100, size=num_samples)
    premiums = 100 + ages * 3 + np.random.normal(0, 10, size=num_samples)  # Example regression line
    data = np.column_stack((ages, premiums))
    return torch.tensor(data, dtype=torch.float32)

# Hyperparameters
noise_dim = 100
feature_dim = 1  # 'age' as feature
predictor_dim = 1  # 'premium' as predictor
num_epochs = 10000
batch_size = 64

# Models
generator = Generator(noise_dim, feature_dim + predictor_dim)
discriminator = Discriminator(feature_dim + predictor_dim)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
torch.autograd.set_detect_anomaly(True)
# Loss functions
adversarial_loss = nn.BCELoss()
regression_loss = nn.MSELoss()

# Training
for epoch in range(num_epochs):
    # Train Discriminator
    real_data = get_real_data(batch_size)
    
    # Real data
    real_labels = torch.ones(batch_size, 1)
    output_real = discriminator(real_data)
    loss_real = adversarial_loss(output_real, real_labels)
    
    # Fake data
    noise = torch.randn(batch_size, noise_dim)
    fake_data = generator(noise)
    
    # Transform fake data ages to be in the range [70, 100]
    fake_data[:, 0] = fake_data[:, 0] * 30 + 70
    
    fake_labels = torch.zeros(batch_size, 1)
    output_fake = discriminator(fake_data.detach())
    loss_fake = adversarial_loss(output_fake, fake_labels)
    
    # Total discriminator loss
    loss_D = (loss_real + loss_fake) / 2
    
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()
    
    # Train Generator
    output_fake = discriminator(fake_data)
    loss_G_adv = adversarial_loss(output_fake, real_labels)
    
    # Regression constraint
    real_features = real_data[:, 0]  # 'age'
    real_predictor = real_data[:, 1]  # 'premium'
    fake_features = fake_data[:, 0]
    fake_predictor = fake_data[:, 1]
    
    # Regression model
    regression_model = lambda x: 100 + x * 3
    predicted_predictor = regression_model(fake_features)
    loss_G_reg = regression_loss(predicted_predictor, fake_predictor)
    
    # Total generator loss
    loss_G = loss_G_adv + loss_G_reg
    
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")

# Generate some synthetic data
noise = torch.randn(1000, noise_dim)
synthetic_data = generator(noise).detach().numpy()
synthetic_data[:, 0] = synthetic_data[:, 0] * 30 + 70  # Transform ages to [70, 100]

# Plot real vs synthetic data
real_data = get_real_data(1000).numpy()

plt.scatter(real_data[:, 0], real_data[:, 1], color='blue', label='Real Data')
plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], color='red', label='Synthetic Data')
plt.xlabel('Age')
plt.ylabel('Premium')
plt.legend()
plt.show() 

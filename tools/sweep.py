import wandb

# Set up your default hyperparameters
hyperparameter_defaults = dict(
    channels=[16, 32],
    batch_size=100,
    learning_rate=0.001,
    optimizer="adam",
    epochs=2,
    )

# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults)
# Access all hyperparameter values through wandb.config
config = wandb.config

# Set up your model
model = make_model(config)

# Log metrics inside your training loop
for epoch in range(config["epochs"]):
    val_acc, val_loss = model.fit()
    metrics = {"validation_accuracy": val_acc,
               "validation_loss": val_loss}
    wandb.log(metrics)
class Config:
    train_params = {
        "epochs": 100,
        "batch_size": 64,
        "validation_split": 0.2
    }

    model_params = {
        "hidden_units": [128, 128],
        "kernel_regularizer": 1e-4,
        "learning_rate": 0.001,
        "loss_function": "mean_absolute_error"
    }

    callbacks = [
        #"PlotLossesKerasTF()",
        "ModelCheckpoint('best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True)",
        "ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.00001)",
        "EarlyStopping(monitor='loss', patience=10)"
    ]

    supported_loss_functions = ["mean_squared_error", "mean_absolute_error"]

    supported_activation_functions = ["relu", "tanh", "sigmoid"]

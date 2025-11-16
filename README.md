To setup Airsim follow this instruction manual:

https://microsoft.github.io/AirSim/build_windows/

The plan is that for each target (0, 1, 2, 3 [Rest, Left-hand, Right-Hand, Foot]) we will train a model to identify the intent based on 45 Columns of eeg data as input.

The output will be a target, which will then be used to control the drone.

Currently, the data is split within the "Dataset/Modified" folder where the %80 will be used for training and validation

and the %20 will be fed into the model to control the drone with.

(0, 1, 2, 3 [Rest, Left-hand, Right-Hand, Foot])

    0: "forward",    Rest

    1: "left",           Left-Hand
    2: "right",         Right- Hand

    3: "backward", Foot

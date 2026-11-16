# Sensor Noise Robustness & MF/Defuzzification Ablation Study Results

## 1. Sensor Noise (σ_η = 0.2°, σ_ω = 0.35°/s) & External Torque Disturbance Robustness

| Experiment                 | Configuration           | ISE    | ITSE   | Control_Effort_||E|| | J_Index | Settling_Time_s | Chattering_TV | Steady_State_MSE |
| -------------------------- | ----------------------- | ------ | ------ | -------------------- | ------- | --------------- | ------------- | ---------------- |
| Sensor_Noise_&_Disturbance | STA                     | 16.183 | 59.773 | 0.4911               | 0.5551  | 38.5            | 32.003        | 0.2706           |
| Sensor_Noise_&_Disturbance | ASTA                    | 15.039 | 51.621 | 0.6811               | 0.5822  | 18.6            | 77.232        | 0.2516           |
| Sensor_Noise_&_Disturbance | STA-Fuzzy-Type1-GA      | 16.599 | 62.579 | 0.4149               | 0.5414  | 38.25           | 34.738        | 0.2775           |
| Sensor_Noise_&_Disturbance | STA-Fuzzy-Type2-GA      | 15.195 | 53.646 | 0.4155               | 0.4881  | 38.35           | 35.69         | 0.2542           |
| Sensor_Noise_&_Disturbance | STA-Fuzzy-Type2-Taguchi | 15.566 | 56.901 | 0.5872               | 0.5763  | 37.95           | 31.81         | 0.2603           |
| Sensor_Noise_&_Disturbance | Proposed-AGS-IT2FLC     | 15.082 | 52.75  | 0.4121               | 0.4814  | 38.1            | 4.056         | 0.2523           |

## 2. Membership Function Prototype & Defuzzification Method Ablation Study

| Experiment               | Configuration                  | ISE    | ITSE   | Control_Effort_||E|| | J_Index | Settling_Time_s | Chattering_TV | Steady_State_MSE |
| ------------------------ | ------------------------------ | ------ | ------ | -------------------- | ------- | --------------- | ------------- | ---------------- |
| MF_Prototype_Ablation    | Type-1 (GA, L=3) - Gaussian    | 15.034 | 51.37  | 0.3283               | 0.4395  | 21.45           | 1.695         | 6.257e-11        |
| MF_Prototype_Ablation    | Type-2 (GA, L=3) - Gaussian    | 14.173 | 45.854 | 0.3579               | 0.4183  | 21.45           | 1.809         | 6.240e-11        |
| MF_Prototype_Ablation    | Type-1 (GA, L=3) - Triangular  | 14.381 | 47.286 | 0.3542               | 0.4254  | 21.65           | 1.72          | 6.311e-11        |
| MF_Prototype_Ablation    | Type-2 (GA, L=3) - Triangular  | 13.974 | 45.047 | 0.3309               | 0.4026  | 21.05           | 1.675         | 6.103e-11        |
| MF_Prototype_Ablation    | Type-1 (GA, L=3) - Trapezoidal | 14.394 | 47.394 | 0.3499               | 0.4243  | 21.55           | 1.712         | 6.265e-11        |
| MF_Prototype_Ablation    | Type-2 (GA, L=3) - Trapezoidal | 13.912 | 44.624 | 0.3346               | 0.4016  | 21.1            | 1.675         | 6.125e-11        |
| Defuzzification_Ablation | Type-1 GA - CENTROID           | 15.034 | 51.37  | 0.3283               | 0.4395  | 21.45           | 1.695         | 6.257e-11        |
| Defuzzification_Ablation | Type-1 GA - BISECTOR           | 14.558 | 48.406 | 0.3434               | 0.4278  | 21.45           | 1.705         | 6.245e-11        |
| Defuzzification_Ablation | Type-1 GA - MOM                | 14.427 | 48.246 | 0.3861               | 0.4439  | 22.9            | 1.769         | 6.993e-11        |
| Defuzzification_Ablation | Type-1 GA - SOM                | 14.432 | 47.908 | 0.3692               | 0.4351  | 22.3            | 1.741         | 6.733e-11        |

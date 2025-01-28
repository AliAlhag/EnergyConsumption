## Dataset Overview

The dataset contains the following features:
- `RoomArea`: The area of the room in square meters.
- `NumberofAppliances`: Number of appliances in the room.
- `Outside Temperature`: Temperature outside the building in Celsius.
- `InsulationThickness`: Thickness of insulation in centimeters.
- `BuildingType`: The type of building (Residential/Commercial).
- `HVACSystem`: Type of HVAC system (e.g., Central AC, Window AC, etc.).
- `AverageTemperature`: Average temperature inside the building.
- `EnergyConsumption`: Target variable representing energy consumption in kilowatt-hours (kWh).

---

## Workflow

1. **Preprocessing**:
   - Encode categorical variables (`BuildingType`, `HVACSystem`) using `LabelEncoder`.
   - Standardize numerical features for better model performance.

2. **Model Training**:
   - Split the dataset into training (60%), validation (20%), and test (20%) sets.
   - Train a **LightGBM Regressor** with the following parameters:
     - `n_estimators=500`
     - `learning_rate=0.05`
     - `max_depth=5`
   - Evaluate the model on validation and test sets.

3. **Evaluation Metrics**:
   - **MAE (Mean Absolute Error)**
   - **MSE (Mean Squared Error)**
   - **R² Score**

4. **Visualization**:
   - A bar plot compares R² scores for the validation and test datasets.



# ML-Techniques-2020-Final-Project
## Topic
Revenue prediction: Ordinal ranking w/ error function L1  
Hotel booking demand dataset  
https://www.kaggle.com/jessemostipak/hotel-booking-demand

## Requirements
Compare least three ML approaches (pros and cons)  
Preprocess details, experimental settings  
For for detail
https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/project/project.pdf

## Observations
119k data are divided into training 90k and the rest testing  
Entries to be predicted: is_canceled( up to 37%), adr, reservation_status_date x reservation_status  

## Implementation Details
### Model
- is_cancel: classification; ADR: regression  
- is_cancel: regression; ADR: regression (BASELINE)
- is_cancel * ADR: regression  

### Catagories
- NN (pytorch)  
- Boosted tree (XGBoost)  
- Decision tree (sklearn)  
- // Conventional regression (sklearn)  

### Preprocessing
如果有缺失值的 rows，可以直接drop掉。  
依feature的variance去決定feature重要性，重要的話就把feature取高次項  
Techniques  
- Standardization: x’ = (x - min)/(max - min) (features with value correlation)
    - Add x^2, x^3, … (depends on the variance?)
- Proper one-hot encoding (features with no value correlation)
- Special entries (previous_bookings_not_canceled) modified to 0 / 1
- Duplicated Entries Removal

### NN-Model
- Suffle dataset -> K-fold validation  
- 2-4 hidden layers to prevent overfitting  
- neuron at each layer < features  
- different activation functions  

## Schedule
- 12/18: Baseline of regression of each method  
- 12/25: Primary statistics, determine the best, confirm preprocessing method, confirm experiments to do  
- 1/1: Tables completed  
- 1/8: Buffer  
- 1/11: Competition ends at noon  
- 1/15 Report completed  
- 1/19: Report due at noon   

# Feature preprocessing lists

- oheList = ['hotel',  'meal', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type']  

- standardizeList = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list', 'required_car_parking_spaces', 'total_of_special_requests']  

- donothingList = ['is_repeated_guest']  

- dropList = ['ID', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month', 'country', 'agent', 'company', 'reservation_status', 'reservation_status_date']

- labelList = ['is_canceled', 'adr']  


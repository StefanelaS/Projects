import faust
import json
from datetime import datetime

# Defining Faust app
app = faust.App('temperature-stream', broker='kafka://localhost:29092')

# Defining the input and output Kafka topics
input_topic = app.topic('InputData', value_type=bytes)
output_topic = app.topic('Output', value_type=bytes)
outliers_topic = app.topic('Outliers', value_type=bytes)

# Defining Faust schemas for input and output data
class InputRecord(faust.Record):
    WBANNO: str
    UTC_DATE: int
    UTC_TIME: int
    LST_DATE: int
    LST_TIME: int
    CRX_VN: str
    LONGITUDE: float
    LATITUDE: float
    AIR_TEMPERATURE: float
    PRECIPITATION: float
    SOLAR_RADIATION: float
    SR_FLAG: str
    SURFACE_TEMPERATURE: float
    ST_TYPE: str
    ST_FLAG: str
    RELATIVE_HUMIDITY: float
    RH_FLAG: str
    SOIL_MOISTURE_5: float
    SOIL_TEMPERATURE_5: float
    WETNESS: float
    WET_FLAG: str
    WIND_1_5: float
    WIND_FLAG: str
    source_file: str

class HourlyTemperature(faust.Record):
    station: str
    lst_date: str
    lst_hour: str
    mean_temp: float

class OutlierRecord(faust.Record):
    station: str
    lst_date: int
    lst_time: int
    temp: float
    source_file: str

# Function for detecting outliers
def detect_outliers(record):
    if record.AIR_TEMPERATURE < -60 or record.AIR_TEMPERATURE > 60:
        outlier = OutlierRecord(
            station=record.WBANNO,
            lst_date=f"{str(record.LST_DATE)[:4]}-{str(record.LST_DATE)[4:6]}-{str(record.LST_DATE)[6:]}",
            lst_time=f"{str(record.LST_TIME)[:2]}:{str(record.LST_TIME)[2:]}",
            temp=record.AIR_TEMPERATURE,
            source_file = record.source_file
        )
        return outlier
    return None

# Function for computing mean temperature 
def compute_mean_temp(station, date, hour, temps):
    mean_temp = sum(temps) / len(temps)
    hourly_temp = HourlyTemperature(
        station=station,
        lst_date=f"{str(date)[:4]}-{str(date)[4:6]}-{str(date)[6:]}",
        lst_hour=hour,
        mean_temp=mean_temp
    )
    return hourly_temp


@app.agent(input_topic)
async def process_temperature(records):
    current_date = None
    current_hour = None
    current_station = None
    temps = []
    async for record in records:
        record = InputRecord(**record)

        # Check if record is an outlier and skip the rest of the loop if it is
        outlier = detect_outliers(record)
        if outlier:
            await outliers_topic.send(value=outlier)
            continue
        
        station = record.WBANNO
        date = record.LST_DATE
        lst_time = f"{record.LST_TIME:04d}"
        hour = lst_time[:2]
        temp = record.AIR_TEMPERATURE

        # Check if the hour or station has changed and compute hourly mean temperature
        if (current_station is not None and (hour != current_hour or date != current_date)):
            hourly_temp = compute_mean_temp(current_station, current_date, current_hour, temps)
            await output_topic.send(value=hourly_temp)
            temps = []

        # Update current hour, date and station
        current_hour = hour
        current_station = station
        current_date =  date

        # Append the temperature for current record to the list
        temps.append(temp)

    hourly_temp = compute_mean_temp(current_station, current_date, current_hour, temps)
    await output_topic.send(value=hourly_temp)


# Defining the input and output Kafka topics
partitioned_topic = app.topic('Partitioned', value_type=bytes)
max_temp_topic = app.topic('MaxTemp', value_type=bytes)

# Defining Faust Schema for the output
class MaxTemperatureRecord(faust.Record):
    max_temp: float
    station: str
    lst_date: str
    lst_hour: str
    source_file: str

# Dictionary to hold the current records for comparison
current_records = {0: [], 1: [],2: []}

@app.agent(partitioned_topic)
async def max_temp_process(stream):
    async for event in stream.events():
        record = InputRecord(**event.value)
        partition = event.message.partition

        current_records[partition].append(record)

        # Check if we have 12 records for each partition
        if all(len(records) >= 12 for records in current_records.values()):
            combined_records = [rec for recs in current_records.values() for rec in recs[:12]]

            # Find the record with the maximum temperature
            max_record = max(combined_records, key=lambda x: x.AIR_TEMPERATURE)

            # Create the max temperature record
            date = max_record.LST_DATE
            lst_time = f"{record.LST_TIME:04d}"
            max_temp_record = MaxTemperatureRecord(
                max_temp=max_record.AIR_TEMPERATURE,
                station=max_record.WBANNO,
                lst_date=f"{str(date)[:4]}-{str(date)[4:6]}-{str(date)[6:]}",
                lst_hour=lst_time[:2],
                source_file=max_record.source_file
            )
            await max_temp_topic.send(value=max_temp_record)


            # Reset the current_records dictionary
            for partition in current_records.keys():
                current_records[partition] = current_records[partition][12:]


if __name__ == '__main__':
    app.main()

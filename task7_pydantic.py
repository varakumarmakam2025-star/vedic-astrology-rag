from pydantic import BaseModel, ValidationError

class ChartReading(BaseModel):
    sign: str
    house: int
    birth_date: str


reading = ChartReading(sign="Leo", house=5, birth_date="1998-08-10")
print(reading)
print(reading.sign)
print(reading.house)

try:
    bad_reading = ChartReading(sign="Leo", house="fifth", birth_date="1998-08-10")
    print(bad_reading)
except ValidationError:
    print("Invalid chart data provided!")
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver

# Initialize the ChromeDriver
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 30)  # Timeout for elements

driver.get("https://www.uber.com/global/en/price-estimate/?_csid=-d7cxXx4Ttsj-2pNNEK04g&effect=&state=Fn-RI93pfMgusnnp3XSdB2pJ16lmuAcfHaLB_Ciot3M%3D&wstate=LTkSkaFc0G4fFkNIX7tQXMD_K-cdE2Fs1awy7Japckc%3D")
#driver.get("https://m.uber.com/go/product-selection?delayed=false&drop%5B0%5D=%7B%22addressLine1%22%3A%22Central%20Railway%20Station%22%2C%22addressLine2%22%3A%22Periamet%2C%20Jutkapuram%2C%20Park%20Town%2C%20Chennai%2C%20Tamil%20Nadu%22%2C%22id%22%3A%22ChIJO2QA7v9lUjoR_e8-D6sQJzk%22%2C%22source%22%3A%22SEARCH%22%2C%22latitude%22%3A13.083397%2C%22longitude%22%3A80.276202%2C%22provider%22%3A%22google_places%22%7D&marketing_vistor_id=3478dea4-5bbb-48d7-ae5b-d316e9d05544&pickup=%7B%22addressLine1%22%3A%22Chennai%20International%20Airport%22%2C%22addressLine2%22%3A%22Airport%20Rd%2C%20Meenambakkam%2C%20Chennai%2C%20Tamil%20Nadu%22%2C%22id%22%3A%22ChIJl2OoXR9eUjoRR27ibiEvCSE%22%2C%22source%22%3A%22SEARCH%22%2C%22latitude%22%3A12.9821555%2C%22longitude%22%3A80.1641598%2C%22provider%22%3A%22google_places%22%7D&redirect=false&uclick_id=ae7e6e90-d515-49b1-b33d-f1025711ce49&vehicle=2019")
driver.maximize_window()

# Wait for the google input field to be clickable
#google_input = wait.until(EC.element_to_be_clickable((By.ID, 'google-login-btn')))
#google_input.click()  # Click the field first

login_cred = driver.find_element(By.ID, "PHONE_NUMBER_or_EMAIL_ADDRESS")
login_cred.send_keys("jonahsingh.davids@gmail.com") 

# Wait for the login input field to be clickable
login_input = wait.until(EC.element_to_be_clickable((By.ID, 'forward-button')))
login_input.click()

'''ride_details=[]
car_items = driver.find_elements(By.XPATH, '//div[contains(@class, "clearfix bus-item")]')  # Selector for bus items
for car_item in car_items:
    # Find elements by class name
    type_element = car_item.find_element(By.CLASS_NAME, "_css-jsRibq")
    passenger_element = car_item.find_element(By.CLASS_NAME, "_css-egaLzu")
    fare_element = car_item.find_element(By.CLASS_NAME, "_css-jeMle")
    pickup_reach_element = car_item.find_element(By.CLASS_NAME, "_css-bNXHBf")
    car_type_element = car_item.find_element(By.CLASS_NAME, "_css-iqMJpM")
    print(f"Type: {type_element},Passenger-no: {passenger_element}, Fare: {fare_element}, Pickup-in & Drop by: {pickup_reach_element}, Car type: {car_type_element}")
    print("*********************************") 
    ride_details.append({
            'Type': type_element,  # Example route name
            'Passenger-no': passenger_element,
            'Fare': fare_element,
            'Pickup-in & Drop by': pickup_reach_element,
            'Car type': car_type_element
        })

# Store scraped data in CSV
ride_df = pd.DataFrame(ride_details)
ride_df=ride_df.drop_duplicates()
csv_file_path = 'Ride_Details.csv'
ride_df.to_csv(csv_file_path, index=False, mode='w', header=True)  # Overwrite the CSV file
print(f"Data saved to {csv_file_path}")'''

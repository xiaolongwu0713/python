import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


class Browser:
    browser, service = None, None

    # Initialise the webdriver with the path to chromedriver.exe
    def __init__(self, driver: str):
        self.service = Service(driver)
        self.browser = webdriver.Chrome(service=self.service)

    def open_page(self, url: str):
        self.browser.get(url)

    def close_browser(self):
        self.browser.close()

    def add_input(self, by: By, value: str, text: str):
        field = self.browser.find_element(by=by, value=value)
        field.send_keys(text)
        time.sleep(1)

    def click_button(self, by: By, value: str):
        button = self.browser.find_element(by=by, value=value)
        button.click()
        time.sleep(1)

    def login_overleaf(self, username: str, password: str):
        self.add_input(by=By.ID, value='email', text=username)
        self.add_input(by=By.ID, value='password', text=password)
        self.click_button(by=By.CLASS_NAME, value='btn-primary') # actual: class="btn-primary btn btn-block"

    def select_all_project(self, by:By, value: str):
        button = self.browser.find_element(by=by, value=value)
        button.click()

    def click_download(self, by:By, value: str):
        button = self.browser.find_element(by=by, value=value)
        button.click()

if __name__ == '__main__':
    browser = Browser('C:/Users/Public/venvs/bci/Scripts/chromedriver.exe')

    browser.open_page('https://www.overleaf.com/login')
    time.sleep(3)

    browser.login_overleaf(username='xiaolongwu0713@gmail.com', password='919420mm')
    time.sleep(5)

    browser.select_all_project(by=By.CLASS_NAME, value='project-list-table-select-item')
    time.sleep(3)

    browser.click_download(by=By.CLASS_NAME, value='fa')
    time.sleep(30)
    browser.close_browser()
"""Script that scrapes the global top 200 charts.

From https://spotifycharts.com/regional/global/daily/.
"""

import pandas as pd
import selenium.webdriver
from selenium.webdriver.firefox.options import Options

# date = str(datetime.strftime(datetime.today(), '%d%m%Y'))
date = "2021-08-30"

# Inintialize browser
options = Options()
options.add_argument("--headless")
browser = selenium.webdriver.Firefox(options=options)

# Initialize url for daily global 200
url_top_200 = f"https://spotifycharts.com/regional/global/daily/{date}"

browser.get(url_top_200)

# with open(f'screenshot.{int(time.time())}.png', 'wb')
# as fp: fp.write(browser.get_screenshot_as_png())

# Select all songs in list as elements
song_elems = browser.find_element_by_xpath(
    "/html/body/div/div/div/div/span/table/tbody/tr"
)

# Extract necessary features from each element (song)
songs_list = []
for elem in song_elems:
    rank = elem.find_element_by_class_name("chart-table-position").text
    url = elem.find_element_by_xpath("//td/a").get_attribute("href")
    songs_list.append({"rank": rank, "url": url})

# Convert to DataFrame and save in csv
df = pd.DataFrame(songs_list).assign(date=date)

df.to_csv(f"../data/top_200_{date}.csv", index=False)

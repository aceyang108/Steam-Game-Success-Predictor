from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import undetected_chromedriver as uc
import pandas as pd
import time, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')

app_ids = []
app_names = []
followers_count = []
rank = []

try:
    # url: "https://steamdb.info/stats/mostwished/" -> 未發售作品的願望清單排名
    # url: "https://steamdb.info/stats/wishlistactivity/" -> 所有作品的願望清單排名（含已發售）
    # 我兩種版本都有抓下來，前者的問題是沒辦法作為 feature 加入主資料集，並且若我們要預測已發售遊戲，此資料集不會產生任何影響；
    # 後者的問題是瀏覽器支援顯示的資料量太少（僅1000筆）。
    # 經過我測試，只有使用 Edge 瀏覽器（已知Chrome, Firefox不行）才能看到更完整的清單，但偏偏 SteamDB 網站會擋 Edge 的 webdriver，試了好幾種方法都繞不過
    url = "https://steamdb.info/stats/mostwished/"
    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options)
    driver.get(url)

    input("Please complete the CAPTCHA in the browser window, then press ENTER in command line.")

    select_element = driver.find_element(By.ID, 'dt-length-0')
    select = Select(select_element)
    select.select_by_value("5000")

    time.sleep(10)

    parent_xpath = '//*[@id="DataTables_Table_0"]/tbody'
    parent_element = driver.find_element(By.XPATH, parent_xpath)

    apps = parent_element.find_elements(By.XPATH, ".//*[@class='app']")
    print(f'# of apps: {len(apps)}')
    for app in apps:
        appid = app.get_attribute('data-appid')
        app_ids.append(appid)
        xpath = f"//tr[@data-appid={appid}]"
        app_names.append(app.find_element(By.XPATH, f'{xpath}/td[3]/div/a').text)
        followers_count.append(app.find_element(By.XPATH, f'{xpath}/td[8]').get_attribute('data-sort'))
        rank.append(app.find_element(By.XPATH, f'{xpath}/td[1]').get_attribute('data-sort'))

except Exception as e:
    print(e)
finally:
    driver.quit()

'''print(f'len of rank: {len(rank)}')
print(f'len of appid: {len(app_ids)}')
print(f'len of followers_count: {len(followers_count)}')'''

df_wishlist = pd.DataFrame()
df_wishlist['rank'] = rank
df_wishlist['appid'] = app_ids
df_wishlist['name'] = app_names
df_wishlist['followers'] = followers_count

remove_condition = df_wishlist['name'].str.contains('SteamDB Unknown App', na=False)
df_wishlist = df_wishlist[~remove_condition]

save_path = os.path.join(PROCESSED_DATA_PATH, 'wishlists.csv')
df_wishlist.to_csv(save_path, index=False)
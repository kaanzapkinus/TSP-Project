import pandas as pd
import matplotlib.pyplot as plt
# Dosyayı aç
infile = open('berlin11_modified.tsp', 'r')

# Başlık bilgisini oku
Name = infile.readline().strip().split()[1]  # NAME
FileType = infile.readline().strip().split()[1]  # TYPE
Comment = infile.readline().strip().split()[1]  # COMMENT
Dimension = infile.readline().strip().split()[1]  # DIMENSION
EdgeWeightType = infile.readline().strip().split()[1]  # EDGE_WEIGHT_TYPE
infile.readline()  # Boş satır atla

# Şehir bilgilerini oku
nodelist = []
N = int(Dimension)  # Şehir sayısını sayısal değere çevir
for i in range(N):
    line = infile.readline().strip().split()  
    city_id = int(line[0])  # Şehir ID'si
    x = float(line[1])  # X koordinatı
    y = float(line[2])  # Y koordinatı
    nodelist.append([city_id, x, y])  # Şehir bilgilerini listeye ekle


# Veriyi pandas DataFrame'e yükle
df = pd.DataFrame(nodelist, columns=['City_ID', 'X_Location', 'Y_Location'])
# Şehir bilgilerini yazdır
print(df)

# Şehirlerin konumlarını görselleştirelim
plt.figure(figsize=(10, 6))
plt.scatter(df['X_Location'], df['Y_Location'], color='blue', label='Cities')

# Şehirlerin ID'lerini grafikte göstermek için
for i, row in df.iterrows():
    plt.text(row['X_Location'] + 0.1, row['Y_Location'] + 0.1, str(row['City_ID']), fontsize=8)

# Başlık, etiketler ve grid ekleyelim
plt.title('Cities Location')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()

# Grafiği göster
plt.show()

# Şehir bilgilerini düzenli bir tablo olarak yazdır
print(df)

# Alternatif olarak, veriyi daha rahat görmek için DataFrame'i kaydedebiliriz (isteğe bağlı)
# df.to_csv('cities.csv', index=False)  # CSV dosyası olarak kaydet
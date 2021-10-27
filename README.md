# ScipyTeam_JC_DS_VL_01_FinalProject_NP
Dimas Dwi Setiawan, [email](mailto:s.dimasdwi@gmail.com).

## Telco Customer Churn
Sebuah proyek machine learning yang menggunakan algoritma klasifikasi untuk memprediksi apakah pelanggan akan berhenti atau tidak.

## Dataset
Data bersumber dari [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

Data tersebut merupakan ringkasan dari data [_Telco Customer Churn_](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) milik IBM Business Analytics Community. Data berisi informasi mengenai sebuah perusahaan telekomunikasi fiktif yang menyediakan layanan telepon rumah dan layanan internet kepada 7043 pelanggan di California pada Q3. Data yang terkandung di dalamnya meliputi pelanggan yang tetap bertahan, pelanggan yang berhenti, layanan apa saja yang digunakan oleh pelanggan, demografi pelanggan, dan beberapa informasi penting lainnya.

## Tujuan
Dalam dunia bisnis, terdapat istilah Acquisition Cost dan Retention Cost. Acquisition Cost adalah biaya yang dikeluarkan oleh perusahaan untuk mendapatkan pembeli atau pelanggan baru. Sedangkan Retention Cost adalah biaya yang dikeluarkan oleh perusahaan untuk mempertahankan pelanggan yang ada.

Pada kenyataannya, kita sebagai manusia memiliki keterbatasan dalam memprediksi pelanggan mana yang akan beralih dari produk kita dan mana pelanggan yang akan tetap bertahan. Menurut beberapa sumber, besaran Acquisition Cost 5x lebih besar daripada Retention Cost. Jika kita salah memprediksi pelanggan mana yang sebenarnya akan churn, namun kita prediksi ia akan tetap bertahan, maka biaya yang dikeluarkan menjadi lebih besar.

### Problem Statement for Machine Learning:

1. Berapa banyak keuntungan yang hilang karena pelanggan churn? 
1. Model apa yang paling tepat untuk mendeteksi pelanggan yang akan churn?
1. Bagaimana memprediksi pelanggan yang churn dengan baik, sehingga dapat meminimalisasi prediksi yang berupa false negatif?
1. Matriks apa yang akan digunakan untuk pengukuran kualitas machine learning?

### Problem Statement for Analytics:

1. Customer seperti apa yang paling banyak churn?
1. Apa yang membuat customer bertahan menggunakan provider? Produk unggulan nya apa?
1. Layanan apa yang paling tidak diminati oleh customer?

## Penjelasan Dataset 
Dataset yang ada meliputi `7043` pengamatan dengan `20` fitur dan 1 label (`Churn`)

| No | __Nama Fitur__ | __Penjelasan__ | __Data Type__ |
| - | - | - | - | 
| 1 | CustomerID | Berisi ID unik setiap pelanggan | categorical | 
| 2 | Gender | Apakah pelanggan pria atau wanita | categorical |
| 3 | SeniorCitizen | Apakah pelanggan berusia di atas 65 tahun atau tidak (1, 0) | numeric, int |
| 4 | Partner | Apakah pelanggan memiliki pasangan atau tidak (Yes, No) | categorical |
| 5 | Dependents | Apakah pelanggan memiliki tanggungan atau tidak (Yes, No) | categorical | 
| 6 | Tenure | Jumlah bulan berlangganan pada perusahaan | numeric, int |
| 7 | PhoneService | Apakah pelanggan memiliki layanan telepon atau tidak (Yes, No) | categorical |
| 8 | MultipleLines | Apakah pelanggan menggunakan layanan multiple lines atau tidak (Yes, No, No phone service) | categorical |
| 9 | InternetService | Internet service provider yang digunakan pelanggan (DSL, Fiber optic, No) | categorical |
| 10 | OnlineSecurity | Apakah pelanggan memiliki layanan online security atau tidak (Yes, No, No internet service) | categorical | 
| 11 | OnlineBackup |  Apakah pelanggan memiliki layanan online back up atau tidak (Yes, No, No internet service) | categorical | 
| 12 | DeviceProtection | Apakah pelanggan memiliki layanan device protection atau tidak (Yes, No, No internet service) | categorical |
| 13 | TechSupport | Apakah pelanggan memiliki layanan tech support atau tidak (Yes, No, No internet service) | categorical | 
| 14 | StreamingTV | Apakah pelanggan menggunakan internet untuk streaming TV dari pihak ketiga atau tidak (Yes, No, No internet service) | categorical |
| 15 | StreamingMovies | Apakah pelanggan menggunakan internet untuk streaming film dari pihak ketiga atau tidak (Yes, No, No internet service) | categorical |
| 16 | Contract | Durasi kontrak pembayaran dari pelanggan (Month-to-month, One year, Two year) | categorical |
| 17 | PaperlessBilling | Apakah pelanggan menggunakan paperless billing atau tidak (Yes, No) | categorical |
| 18 | PaymentMethod | Metode pembayaran pelanggan (Electronic check, Mailed check, Bank transfer, Credit card) | categorical |
| 19 | MonthlyCharges | Jumlah tagihan bulanan dari pelanggan |  numeric , int |
| 20 | TotalCharges | Total keseluruhan tagihan dari pelanggan | object |
| 21 | Churn | Apakah pelanggan churn/beralih atau tidak (Yes or No) | categorical |

## Steps

1. Data Understanding
2. Exploratory Data Analysis (EDA)
3. Modeling & Conclusion 

### Data Understanding

1. Terdapat 20 fitur dan 1 label pada dataset.
2. Drop kolom _customerID_, karena tidak dibutuhkan.
3. Terdapat 11 missing value pada kolom _TotalCharges_ yang diakibatkan pelanggan sudah berhenti berlangganan sebelum adanya tagihan. Dengan kata lain, 11 orang tersebut hanya coba-coba saja.

### Ringkasan Exploratory Data Analysis (EDA)

1. Terdapat pemasukan kotor sebesar $2.862.926,9 juta dollar AS yang hilang akibat pelanggan yang churn.
2. Gender: Hampir tidak ada perbedaan perbedaan persentase churn antara pria dan wanita.
3. Senior Citizen: Persentase churn dalam kelompok senior citizen adalah 42%, ada indikasi churn yang tinggi pada kelompok ini.
4. Partner dan Dependents: Kedua fitur memiliki korelasi dan berkontribusi besar dalam kecenderungan pelanggan untuk churn.
5. Phone Service dan Internet Service: Ada sebagian pelanggan yang tidak memiliki internet service dan bahkan sejumlah kecil pelanggan tidak memiliki phone service. Sebagian besar feature lainnya yang tersedia berhubungan dengan internet service.
6. Internet Service: Pelanggan yang menggunakan DSL membayar biaya bulanan yang lebih murah daripada pengguna fiber optic. Pengguna fiber optic juga terlihat cenderung lebih banyak untuk churn.
7. Online Security, Online Backup, Device Protection, Tech Support: Keempat layanan tersebut hanya bisa diakses jika pengguna memiliki layanan internet. Pengguna keempat layanan tersebut cenderung untuk tidak churn. Terutama pengguna Tech Support dan Online Security.
8. Streaming TV dan Movies: Menunjukkan data yang sangat mirip. Baik pengguna maupun tidak, perbedaan pelanggan yang churn hanya sedikit.
9. Contract: Pelanggan yang membayar bulanan lebih banyak yang churn.
10. Paperless Billing: Pelanggan yang menggunakan paperless billing lebih banyak yang churn daripada yang tidak. Hal ini dikarenakan banyak pelanggan yang membayar bulanan juga menggunakan electronic check (bagian dari paperles billing) sebagai metode pembayaran.
11. Tenure: Churn paling banyak terjadi di bulan-bulan awal.
12. Monthly Charges dan Total Charges: Churn paling banyak terjadi di biaya bulanan yang tinggi. Rata rata biaya bulanan untuk pelanggan yang churn lebih rendah daripada yang tidak (karena hanya bayar beberapa bulan saja).

### Modeling

| Target |	Persentase	| 
| - | - | 
| Churn	| 26.54% |	
| Tidak	| 73.46%	| 

Ada 2 cara yang dapat digunakan untuk mengatasi data target (y) yang imbalance:

  1. Setting parameter "(class_weight='balanced')" untuk model yang memilikinya dan masing-masing "random_state" parameter.
  1. Menggunakan metode SMOTE oversampling dan setting "random_state" parameter untuk model yang tidak memiliki "class_weight" parameter.

### Base, Tune and Hyperparameter Tuning Model

RepeatedStratifiedKFold dan GridSearchCV digunakan untuk mencari nilai paramater yang terbaik. Scoring yang digunakan adalah 'recall'.

| Model |  Sebelum  | Setelah |
|:-:|:-:|:-:|
| Logistic Regression | 0.787 | 0.799 |
| Decision Tree Classifier | 0.531 | 0.753 |
| Random Forest Classifier | 0.538 | 0.713 |
| XGBoost Classifier | 0.559 | 0.643 |
| K Nearest Classifier | 0.764 | 0.632 |

Berdasarkan hasil confusion matrix dan waktu perhitungan, kita dapati bahwa performa terbaik terdapat dalam model Decision Tree Classifier. Setelah dilakukan hyperparameter tuning pada model, 'recall' score bertambah drastis, yaitu dari 0.48 menjadi 0.85

### Decision Tree Classifier Confusion Matrix

#### Sebelum Hyperparameter Tuning
![dtc](https://user-images.githubusercontent.com/88280579/139032798-7452b98e-1834-4eec-90f7-79e0d1167f08.png)

#### Setelah Hyperparameter Tuning
![dtc_tuned](https://user-images.githubusercontent.com/88280579/139032956-27751966-992b-4578-a7c5-5319ef086668.png)


### Kesimpulan

Model machine learning yang dikembangkan dapat membantu perusahaan untuk menurunkan biaya dan waktu dalam memprediksi pelanggan mana yang akan berhenti. Sebagaimana disebutkan pada bagian problem stetment, fokus kita adalah false negative rate. Dimana kita ketahui sebelumnya bahwa, biaya yang dikeuarkan untuk memperoleh pelanggan 5 kali lebih besar dari biaya mempertahankan pelanggan yang sudah ada.

Dapat kita simpulkan, dari 100 pelanggan yang diprediksi, machine learning hanya menghasilkan kesalahan berjumlah 9 pelanggan atau 8,55%. Artinya 9 pelanggan tersebut tidak akan mendapat perlakuan khusus dari perusahaan dan akan berhenti berlangganan. Terdapat peningkatan dari model sebelumnya sebelum dituning, yakni dari 100 pelanggan yang diteliti, kesalahan prediksi mencapai 19 orang atau 18,64%.

Namun, yang perlu diingat juga bahwa terdapat konsekuensi lain. Yakni dari 100 pelanggan yang diprediksi akan berhenti berlangganan, ternyata 58,36% nya, atau 59 orang akan mendapatkan perlakuan khusus, tapi ternyata mereka tidak ada kecenderungan untuk churn dan akan tetap berlangganan. Jika dibandingkan model sebelumnya juga terdapat peningkatan dari 50.69% atau hanya 51 orang.

### Business Insight

1. Terdapat pemasukan kotor sebesar $2.862.926,9 juta dollar AS yang hilang akibat pelanggan yang churn.
2. Perusahaan harus lebih memperhatikan:
    - Senior citizen, karena lebih berpotensi untuk churn.
    - Fiber optic, baik dari segi harga maupun kualitas. Karena ada indikasi dapat meningkatkan potensi pelanggan untuk churn.
    - Tech support dan online security sebagai fitur unggulan. Dapat dijadikan sebagai sarana promosi kepada pelanggan yang berpotensi untuk churn.
    - 12 bulan pertama pelanggan mulai menggunakan layanan adalah masa yang penting untuk membuat pelanggan merasa nyaman dengan pelayanan yang diberikan. 
3. Tidak ada perbedaan signifikan bagi pria maupun wanita dalam kecenderungan untuk churn. Strategi apapun yang akan dilakukan dapat berjalan pada kedua gender.

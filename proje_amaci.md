## SOSYAL SORUMLULUK DERSİ VİZE RAPORU

**1. Proje Adı**

Hava Durumu Verileriyle Güneş Enerjisi Üretim Tahmini ve Enerji Verimliliği Öneri Sistemi

**2. Proje Ekibi**
    - Semih DEMİR (222802065)
    - Batu KIZMAZOĞLU (222813001)
**3. Projenin Amacı**

Bu projenin temel amacı, ev tipi güneş panellerinden elde edilen geçmiş üretim verilerini ve hava
durumu faktörlerini analiz etmektir. Bu analiz sonucunda, gelecekteki enerji üretimini yüksek
doğrulukla tahmin edebilen bir makine öğrenimi modeli geliştirilmesi hedeflenmektedir.

Proje, sadece teknik bir tahmin modeli oluşturmanın ötesinde, bu tahminleri son kullanıcı için
eyleme geçirilebilir bilgilere dönüştürmeyi amaçlamaktadır. Sistem, kullanıcıya hangi saat
dilimlerinde enerji üretiminin zirvede olacağını bildirerek, çamaşır makinesi, bulaşık makinesi
veya ütü gibi yüksek enerji tüketen cihazların kullanımını bu verimli saatlere kaydırması için
öneriler sunacaktır.

Bu yaklaşımın nihai hedefi, hanenin enerji verimliliğini maksimize etmek, şebekeden çekilen
enerjiyi azaltmak ve bireysel düzeyde sürdürülebilir tüketim alışkanlıklarının benimsenmesini
teşvik etmektir.

**4. Projenin Sosyal Sorumluluk Gerekçesi**

Bu proje, sosyal sorumluluk kavramını "bireysel eylemlerin toplumsal ve çevresel etkiye
dönüştürülmesi" ilkesi üzerine kurmaktadır.

1. **Sürdürülebilirlik ve Çevre Bilinci:** Projenin temel sosyal sorumluluk gerekçesi,
    yenilenebilir enerji kaynaklarının kullanımını teşvik ederek çevresel etkiyi azaltmaktır.
    Kullanıcılara kendi üretimlerini en verimli nasıl kullanacaklarını göstererek, fosil yakıtlara
    olan bağımlılığı bireysel ölçekte azaltmayı ve karbon ayak izini küçültmeyi hedefler.
2. **Tüketim Alışkanlıklarının Dönüşümü:** Proje, enerji tüketimi konusunda bir "farkındalık"
    yaratmayı amaçlar. Enerjinin ne zaman "yeşil" (doğrudan güneşten üretilen) ve bol
    olduğunu bilmek, tüketicinin davranışını pasif bir "tüketici" olmaktan, enerjiyi "bilinçli
    yöneten" aktif bir "üretici-tüketici" (prosumer) olmaya doğru evrilmesini destekler.
3. **Ekonomik Katkı ve Enerji Tasarrufu:** Sosyal sorumluluk sadece çevresel değil, aynı
    zamanda ekonomiktir. Kullanıcılara enerji tasarrufu sağlamak, aile bütçesine doğrudan
    katkı anlamına gelir. Özellikle enerji maliyetlerinin arttığı dönemlerde, mevcut kaynakları
    (güneş enerjisi) en verimli şekilde kullanmak, ekonomik sürdürülebilirlik için kritik bir
    öneme sahiptir.


**5. Proje Konusuyla İlgili Literatür Araştırması**

Güneş enerjisi üretim tahmini (PV forecasting), enerji şebekelerinin istikrarı, akıllı şebeke
yönetimi ve enerji piyasaları için kritik öneme sahip bir araştırma alanıdır. Literatür, bu tahminleri
yapmak için istatistiksel yöntemlerden fiziksel modellere ve son yıllarda baskın hale gelen
makine öğrenimi (ML) yaklaşımlarına kadar geniş bir yelpazeyi kapsamaktadır.

- **Hava Durumu Parametrelerinin Etkisi:** Güneş enerjisi üretimi, doğası gereği hava
    koşullarına son derece bağımlıdır. Yapılan çalışmalarda, güneşlenme şiddeti (irradiance)
    en önemli parametre olarak öne çıksa da, panel verimliliği üzerinde doğrudan etkisi olan
    **hava sıcaklığı** ve **bulutluluk** (cloud cover) gibi faktörlerin tahmin doğruluğunu önemli
    ölçüde artırdığı tespit edilmiştir (Kalogirou, 2021). Özellikle bulutluluğun ani ve hızlı
    değişimi, üretimde dalgalanmalara yol açmakta ve bu da makine öğrenimi modelleri için
    en zorlu senaryoları oluşturmaktadır (Ahmad & Mourshed, 2019). Projemiz,
    OpenWeatherMap API'sinden elde edilecek bu kritik verileri (sıcaklık, bulutluluk,
    güneşlenme süresi) modelin girdisi olarak kullanarak literatürdeki bu temel bulguyu
    temel almaktadır.
- **Makine Öğrenimi Modellerinin Karşılaştırılması:** Literatürde, kısa vadeli (saatlik veya
    günlük) PV üretim tahmini için çeşitli ML algoritmaları kullanılmıştır. Yapay Sinir Ağları
    (ANN), Destek Vektör Makineleri (SVM) ve özellikle son dönemde yinelenen sinir ağları
    (RNN) ve Uzun Kısa Vadeli Bellek (LSTM) ağları, zaman serisi verilerinin doğasına uygun
    olmaları nedeniyle yüksek başarı göstermektedir (Raza vd., 2020). Projemizde kullanmayı
    planladığımız Scikit-learn ve TensorFlow kütüphaneleri, bu modellerin uygulanması için
    endüstri standardı araçlardır. Bu proje, hangi modelin (örneğin basit bir Lineer
    Regresyon, Random Forest veya daha karmaşık bir Nöral Ağ) elimizdeki yerel ve kısıtlı veri
    seti üzerinde en iyi MAE/RMSE skorunu verdiğini analiz edecektir.
- **Enerji Verimliliği ve Davranışsal Değişim:** Akademik çalışmaların çoğu, şebeke
    operatörleri veya büyük ölçekli santraller için tahmin yapmaya odaklansa da, son
    dönemde "tüketici tarafı yönetimi" (demand-side management) ve "davranışsal enerji
    verimliliği" konuları önem kazanmıştır. Enerji tüketiminin, üretimle senkronize edilmesi
    (yük kaydırma), şebeke üzerindeki stresi azaltır (Gjorgievski vd., 2021). Bu proje,
    literatürdeki bu boşluğu doldurarak, gelişmiş tahmin modellerini doğrudan son
    kullanıcıya "davranışsal bir öneri"^18 olarak sunmayı ve teknik bir aracı sosyal bir
    sorumluluk aracına dönüştürmeyi hedeflemektedir.
**6. Projenin Yenilikçi Yönleri**

Bu proje, mevcut literatürdeki benzer çalışmalardan ve piyasadaki genel uygulamalardan birkaç
temel noktada ayrışmaktadır:

1. **Hiper-Yerel ve Kişiselleştirilmiş Tahmin:** Proje, bölgesel veya ulusal tahminlerin aksine,
    doğrudan bireyin evindeki panelin **gerçek (historik) üretim verilerini** temel alır. Bu,
    panelin açısı, yönelimi, olası gölgelenmeler veya eskimeden kaynaklı verim düşüşleri gibi
    o eve özgü (hiper-yerel) faktörleri dolaylı olarak modelin öğrenmesini sağlar. Bu sayede
    üretilen tahmin, genel bir hava durumu raporundan çok daha isabetli ve kişiselleştirilmiş
    olur.


2. **Tahminden Eyleme Geçiş (Sosyal Boyut):** Birçok akademik proje, en düşük tahmin
    hatasını (RMSE/MAE) elde etmeye odaklanır ve teknik bir sonuç üretir. Projemizin temel
    yeniliği, bu teknik çıktıyı (örn: "13:00'te 4.5 kWh üretim") almak ve bunu son kullanıcının
    anlayacağı **doğrudan bir eylem önerisine** ("Saat 11:00–14:00 arası çamaşır makinesi
    kullanmak en verimlidir") dönüştürmesidir.
3. **Makine Öğreniminin Demokratikleşmesi:** Proje, Python, Scikit-learn ve API'lar gibi
    güçlü teknolojileri kullanarak, karmaşık veri bilimi süreçlerini (veri toplama, işleme,
    modelleme) bireysel bir sosyal sorumluluk hedefi için kullanmaktadır. Bu, "enerji
    farkındalığını artırma" hedefini soyut bir kavram olmaktan çıkarıp, veriye dayalı somut bir
    araca dönüştürür.
**7. Kaynakça**
- Ahmad, Z., & Mourshed, M. (2019). Impact of cloud cover on solar energy generation and
a case study of the UK. _Journal of Cleaner Production_ , 230, 893-903.
- Gjorgievski, V. Z., Baneshi, M., & Stankoski, S. (2021). Demand-side management in
smart grids: An overview of models and methods. _Energy Reports_ , 7, 2108-2125.
- Kalogirou, S. A. (2021). Solar energy engineering: Processes and systems (3rd ed.).
Academic Press.
- Raza, M. Q., Nadarajah, M., & Ekanayake, J. (2020). A comprehensive review on short-
term solar power forecasting using machine learning techniques. _Journal of Renewable
and Sustainable Energy_ , 12(6), 062701.



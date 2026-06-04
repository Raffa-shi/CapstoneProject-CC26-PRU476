import re
import string
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# Inisialisasi Stemmer dan Stopword
factory_stemmer    = StemmerFactory()
stemmer            = factory_stemmer.createStemmer()

factory_stop       = StopWordRemoverFactory()
sastrawi_stopwords = set(factory_stop.get_stop_words())

custom_stopwords = {
    "yg","ga","gak","nggak","ngga","udah","udh","sdh","tdk",
    "tp","jg","klo","klu","kalo","dgn","krn","utk",
    "bisa","min","kak","mas","mba","mbak","pak","bu",
    "iya","ya","lah","deh","dong","sih","nih","kan",
    "mau","ada","tidak","si","hide","replies","reply","all",
    "bpjs","kesehatan","jkn",
    "halo","hai","hallo","assalamualaikum","selamat",
    "pagi","siang","sore","malam",
    "mohon","tolong","bantu","tanya","info","tahu","tau",
    "kami","kita","mereka","dia",
}

all_stopwords = sastrawi_stopwords.union(custom_stopwords)

normalization_dict = {
    "gak":"tidak","ga":"tidak","nggak":"tidak","ngga":"tidak",
    "udah":"sudah","udh":"sudah","sdh":"sudah",
    "tp":"tapi","krn":"karena","karna":"karena",
    "yg":"yang","dgn":"dengan","utk":"untuk",
    "hrs":"harus","bs":"bisa","blm":"belum",
    "sy":"saya","aq":"saya","ak":"saya",
    "km":"kamu","lu":"kamu","lo":"kamu",
    "klo":"kalau","klu":"kalau","kalo":"kalau",
    "dr":"dari","sm":"sama","aja":"saja","doang":"saja",
    "bgt":"banget","bngt":"banget","jg":"juga",
    "lg":"sedang","lagi":"sedang","skrg":"sekarang",
    "emg":"memang","emang":"memang","gimana":"bagaimana",
    "gmn":"bagaimana","knp":"kenapa",
    "rs":"rumah sakit","faskes":"fasilitas kesehatan",
}

def full_preprocessing(teks: str) -> str:
    """
    Pipeline preprocessing identik dengan notebook training.
    Wajib sama persis agar tidak terjadi training-serving skew.
    """
    teks = str(teks).lower()
    teks = re.sub(r"https?://\S+|www\.\S+", "", teks)
    teks = re.sub(r"@\w+", "", teks)
    teks = re.sub(r"#\w+", "", teks)

    emoji_pat = re.compile(
        "[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF" u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251" u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff" u"\u2640-\u2642"
        u"\u2600-\u2B55" u"\u200d\u23cf\u23e9\u231a\ufe0f\u3030" "]+",
        flags=re.UNICODE
    )
    teks = emoji_pat.sub("", teks)
    teks = re.sub(r"\d+", "", teks)
    teks = teks.translate(str.maketrans("", "", string.punctuation))
    teks = re.sub(r"\s+", " ", teks).strip()

    tokens = teks.split()
    tokens = [normalization_dict.get(t, t) for t in tokens]
    teks   = " ".join(tokens)

    tokens = word_tokenize(teks)
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)
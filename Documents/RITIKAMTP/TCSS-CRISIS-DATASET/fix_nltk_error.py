
# Script dùng để fix lỗi punkt_tab trong Colab
# Copy dòng này vào ô code tiếp theo trong Colab và chạy nó
import nltk
try:
  nltk.data.find('tokenizers/punkt_tab')
except LookupError:
  nltk.download('punkt_tab')

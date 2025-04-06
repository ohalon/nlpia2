from PyPDF2 import PdfFileReader

from nlpia2.constants import DATA_DIR

# side-by-side partially unredacted versions of the report
url_quinta_jurecic1 = 'https://assets.documentcloud.org/documents/6979584/Volume-I-Final.pdf'
url_quinta_jurecic2 = 'https://assets.documentcloud.org/documents/6979583/Volume-II-FINAL.pdf'
stream = open(DATA_DIR / 'The-Mueller-Report.pdf', 'rb')
reader = PdfFileReader(stream)

# muellerPages = []
# for pageNum in range(reader.numPages):
#     muellerPages.append([par for par in mueller1Reader.getPage(pageNum).extractText().split('\n')])
# mueller1.close()

# muellerParagraphs = sum(muellerPages, [])  # flatten the list of lists
# len(muellerParagraphs)
# asciitext = unicode2ascii(text)
# asciitext
# ords = pd.Series([ord(c) for c in asciitext])
# pd.Series(list(asciitext))[(ords > 128)]

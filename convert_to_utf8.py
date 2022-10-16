import codecs
encoded_text = open('alignment', 'rb').read()    #you should read in binary mode to get the BOM correctly
bom = codecs.BOM_UTF16_LE                                      #print dir(codecs) for other encodings
assert encoded_text.startswith(bom)                           #make sure the encoding is what you expect, otherwise you'll get wrong data
encoded_text = encoded_text[len(bom):]                         #strip away the BOM
decoded_text = encoded_text.decode('utf-16le')                 #decode to unicode


file = codecs.open("alignment_utf_8_test", "w", "utf-8")
file.write(decoded_text)
file.close()

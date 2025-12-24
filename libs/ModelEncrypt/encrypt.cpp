#include "encrypt.h"


std::string Encrypt(const std::string& data, const CryptoPP::SecByteBlock& key, const CryptoPP::SecByteBlock& iv) {
    std::string cipher;

    try {
        CryptoPP::GCM<CryptoPP::AES>::Encryption e;
        e.SetKeyWithIV(key, key.size(), iv, iv.size());

        CryptoPP::StringSource(data, true, 
            new CryptoPP::AuthenticatedEncryptionFilter(e,
                new CryptoPP::StringSink(cipher)
            ) // StreamTransformationFilter
        ); // StringSource
    }
    catch(const CryptoPP::Exception& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return cipher;
}

std::string Decrypt(const std::string& cipher, const CryptoPP::SecByteBlock& key, const CryptoPP::SecByteBlock& iv) {
    std::string recovered;

    try {
        CryptoPP::GCM<CryptoPP::AES>::Decryption d;
        d.SetKeyWithIV(key, key.size(), iv, iv.size());

        // The StreamTransformationFilter removes
        //  padding as required.
        CryptoPP::StringSource(cipher, true, 
            new CryptoPP::AuthenticatedDecryptionFilter(d,
                new CryptoPP::StringSink(recovered)
            ) // StreamTransformationFilter
        ); // StringSource

    }
    catch(const CryptoPP::Exception& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return recovered;
}

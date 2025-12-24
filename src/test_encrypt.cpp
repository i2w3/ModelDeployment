#include "encrypt.h"
#include "mac.hpp"


std::string GenerateAESKey(const std::string& content) {
    CryptoPP::byte hash[CryptoPP::SHA256::DIGESTSIZE];
    CryptoPP::SHA256().CalculateDigest(hash, reinterpret_cast<const CryptoPP::byte*>(content.c_str()), content.length());

    CryptoPP::HexEncoder encoder;
    std::string encodedHash;
    encoder.Attach(new CryptoPP::StringSink(encodedHash));
    encoder.Put(hash, sizeof(hash));
    encoder.MessageEnd();

    return encodedHash.substr(0, 32); // AES-256 key length is 32 bytes
}


std::string GenerateRandomIV() {
    CryptoPP::byte iv[CryptoPP::AES::BLOCKSIZE];
    CryptoPP::AutoSeededRandomPool prng;
    prng.GenerateBlock(iv, sizeof(iv));

    CryptoPP::HexEncoder encoder;
    std::string encodedIV;
    encoder.Attach(new CryptoPP::StringSink(encodedIV));
    encoder.Put(iv, sizeof(iv));
    encoder.MessageEnd();

    return encodedIV;
}


int main(){
    std::string mac = GetMACAddress();
    std::string macKey = GenerateAESKey(mac);
    std::string ivStr = GenerateRandomIV();
    std::cout << "MAC Address: " << mac << std::endl;

    CryptoPP::SecByteBlock key(reinterpret_cast<const CryptoPP::byte *>(macKey.data()), macKey.size());
    CryptoPP::SecByteBlock iv(reinterpret_cast<const CryptoPP::byte *>(ivStr.data()), ivStr.size());

    std::string data = "Hello, World!";
    std::string encrypted = Encrypt(data, key, iv);
    std::cout << "Encrypted Data: " << encrypted << std::endl;

    std::string decrypted = Decrypt(encrypted, key, iv);
    std::cout << "Decrypted Data: " << decrypted << std::endl;
    return 0;
}
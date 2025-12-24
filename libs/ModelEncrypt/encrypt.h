#ifndef ENCRYPT_H
#define ENCRYPT_H

#include <string>
#include <iostream>

#include <cryptopp/hex.h>
#include <cryptopp/gcm.h>
#include <cryptopp/osrng.h>

std::string Encrypt(const std::string& data, const CryptoPP::SecByteBlock& key, const CryptoPP::SecByteBlock& iv);

std::string Decrypt(const std::string& cipher, const CryptoPP::SecByteBlock& key, const CryptoPP::SecByteBlock& iv);

#endif // ENCRYPT_H
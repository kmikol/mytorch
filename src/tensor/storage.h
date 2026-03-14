#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>

struct Storage {

    float* data;
    size_t size;


    // Allocating constructor
    // ----------------------
    // The colon after the parameter list begins the "member initializer list".
    // This is the preferred way to initialize members — it runs before the
    // constructor body and is more efficient than assigning inside the body.
    // Here we initialize size_ directly from the parameter.
    Storage(size_t size) : size(size) {
        // Round up the byte count to the nearest multiple of 32.
        // aligned_alloc requires the size to be a multiple of the alignment.
        // e.g. 3 floats = 12 bytes → rounds up to 32 bytes.
        size_t bytes = ((size * sizeof(float) + 31) / 32) * 32;

        // aligned_alloc(alignment, size) allocates memory guaranteed to start
        // at an address divisible by `alignment`. 32-byte alignment satisfies
        // AVX SIMD requirements. We cast the void* it returns to float*.
        data = static_cast<float*>(::aligned_alloc(32, bytes));

        // aligned_alloc returns nullptr on failure (out of memory etc.).
        // We throw std::bad_alloc to signal allocation failure, consistent
        // with how standard containers behave.
        if (!data) throw std::bad_alloc();
    }

    // Destructor
    // ----------
    // Called automatically when the Storage object goes out of scope or is
    // deleted. Responsible for releasing the memory we allocated.
    // The null check guards against the case where data_ was stolen by a
    // move operation and set to nullptr — freeing nullptr is undefined behavior.
    ~Storage() {
        if (data) ::free(data);
    }

    // Copy constructor (deleted)
    // --------------------------
    // The syntax `= delete` tells the compiler to reject any attempt to copy
    // a Storage. Without this, the compiler would generate a default copy
    // constructor that copies the pointer value — resulting in two Storage
    // objects thinking they own the same memory, causing a double free.
    Storage(const Storage& other) = delete;

    // Copy assignment (deleted)
    // -------------------------
    // Same reasoning as copy constructor. Covers the case:
    //   Storage a(1024);
    //   Storage b(1024);
    //   b = a;  // this would use copy assignment — we want this to fail
    Storage& operator=(const Storage&) = delete;

    // Move constructor
    // ----------------
    // `Storage&&` is an "rvalue reference" — it binds to temporary objects or
    // objects explicitly cast with std::move(). The double ampersand distinguishes
    // it from a regular (lvalue) reference `Storage&`.
    // `noexcept` promises this will never throw. This is important — std::vector
    // and other containers will only use the move constructor (instead of copy)
    // if it is marked noexcept.
    Storage(Storage&& other) noexcept
        // Steal the pointer and size from `other` via the initializer list.
        // This copies the pointer value (memory address), not the data itself.
        : data(other.data), size(other.size) {
        // Null out the source so its destructor does not free memory we now own.
        // Without this, both destructors would call free() on the same address.
        other.data = nullptr;
        other.size = 0;
    }

    // Move assignment operator
    // ------------------------
    // Covers the case where an already-constructed Storage is assigned from
    // an rvalue:
    //   Storage a(1024);
    //   Storage b(1024);
    //   b = std::move(a);
    // Unlike the move constructor, `*this` already owns data, so we must
    // release it before stealing from `other`.
    Storage& operator=(Storage&& other) noexcept {
        // Self-assignment guard: if someone writes `a = std::move(a)`,
        // we must not free our own data before stealing it.
        if (this != &other) {
            ::free(data);          // release currently owned memory
            data = other.data;   // steal the pointer
            size = other.size;
            other.data = nullptr; // disarm the source destructor
            other.size = 0;
        }
        // Return a reference to *this so assignment can be chained:
        // a = b = std::move(c);
        return *this;
    }


    // Accessor for data
    // Operator[] provides array-like access to the elements. It does not perform bounds checking.
    // On the other hand, the at() method provides bounds-checked access, throwing an exception if the index is out of range.
    float& operator[](size_t index) { return data[index]; }

    const float& operator[](size_t index) const { return data[index]; }
    
    float& at(size_t index) {
        if (index >= size) throw std::out_of_range("Index out of range");
        return data[index];
    }

    const float& at(size_t index) const {
        if (index >= size) throw std::out_of_range("Index out of range");
        return data[index];
    }

    // Fill the storage with a specific value. Useful for Tensor::zeros and Tensor::ones.
    void fill(float value) {
        std::fill(data, data + size, value);
    }

};
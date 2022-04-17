#ifndef SPH_FUGUE_H__
#define SPH_FUGUE_H__

#include <stddef.h>
#include "sph_types.h"
#include <x86intrin.h>

#ifdef __cplusplus
extern "C"{
#endif

typedef unsigned char BitSequence;
typedef unsigned long long DataLength;
typedef enum { SUCCESS = 0, FAIL = 1, BAD_HASHBITLEN = 2} HashReturn;

typedef unsigned char uint8;
typedef unsigned int uint32;
typedef unsigned long long uint64;

//typedef struct {
//    uint32 buffer[8]; /* Buffer to be hashed */
//    __m128i chainv[10];   /* Chaining values */
//    uint64 bitlen[2]; /* Message length in bits */
//    uint32 rembitlen; /* Length of buffer data to be hashed */
//    int hashbitlen;
//} hashState_luffa;

typedef unsigned char byte;

#define m128_zero      _mm_setzero_si128()

#define mm128_xor3( a, b, c ) \
   _mm_xor_si128( a, _mm_xor_si128( b, c ) )

#define mm128_xim_32( v1, v2, c ) \
   _mm_castps_si128( _mm_insert_ps( _mm_castsi128_ps( v1 ), \
                                    _mm_castsi128_ps( v2 ), c ) )

static inline __m128i mm128_mask_32( const __m128i v, const int m ) 
{   return mm128_xim_32( v, v, m ); }


#ifdef __GNUC__
#define MYALIGN __attribute__((aligned(16)))
#else
#define MYALIGN __declspec(align(16))
#endif

#define M128(x) *((__m128i*)x)


//typedef unsigned char BitSequence;
//typedef unsigned long long DataLength;
//typedef enum {SUCCESS = 0, FAIL = 1, BAD_HASHBITLEN = 2} HashReturn;


//#include "simd-utils.h"


typedef struct
{
	#ifndef DOXYGEN_IGNORE
	__m128i			state[12];
	unsigned int	base;

	unsigned int	uHashSize;
	unsigned int	uBlockLength;
	unsigned int	uBufferBytes;
	DataLength		processed_bits;
	BitSequence		buffer[4];
	#endif

} sph_fugue_context __attribute__ ((aligned (64)));

typedef sph_fugue_context sph_fugue512_context;



// These functions are deprecated, use the lower case macro aliases that use
// the standard interface. This will be cleaned up at a later date.
HashReturn fugue512_Init(sph_fugue512_context *state, int hashbitlen);

HashReturn fugue512_Update(sph_fugue512_context *state, const void *data, DataLength databitlen);

HashReturn fugue512_Final(sph_fugue512_context *state, void *hashval);

HashReturn fugue512_full(sph_fugue512_context *hs, void *hashval, const void *data, DataLength databitlen);

void sph_fugue512_init(void *cc);
void sph_fugue512(void *cc, const void *data, size_t len);
void sph_fugue512_close(void *cc, void *dst);

#ifdef __cplusplus
}
#endif

//#endif // AES
#endif


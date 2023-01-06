#include <stddef.h>
#include "sph_types.h"
#include <x86intrin.h>

#ifdef __cplusplus
extern "C"{
#endif

//#include "algo/sha/sha3_common.h"

//#include <emmintrin.h>

typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

#ifdef __GNUC__
#define MYALIGN __attribute__((aligned(16)))
#else
#define MYALIGN __declspec(align(16))
#endif

#define M128(x) *((__m128i*)x)

#define m128_zero      _mm_setzero_si128()

#define m128_const_64( hi, lo ) \
   _mm_insert_epi64( mm128_mov64_128( lo ), hi, 1 )

#define mm128_xor3( a, b, c ) \
   _mm_xor_si128( a, _mm_xor_si128( b, c ) )

static inline __m128i mm128_mov64_128( const uint64_t n )
{
  __m128i a;
#if defined(__AVX__)
  asm( "vmovq %1, %0\n\t" : "=x"(a) : "r"(n) );
#else
  asm( "movq %1, %0\n\t" : "=x"(a) : "r"(n) );
#endif
  return  a;
}

typedef struct
{
	__m128i			state[4][4];
        BitSequence             buffer[192];
	__m128i			k;
	__m128i			hashsize;
	__m128i			const1536;

	unsigned int	uRounds;
	unsigned int	uHashSize;
	unsigned int	uBlockLength;
	unsigned int	uBufferBytes;
	DataLength		processed_bits;

} sph_echo512_context __attribute__ ((aligned (64)));

void init_echo(sph_echo512_context *state, int hashbitlen);

void reinit_echo(sph_echo512_context *state);

void update_echo(sph_echo512_context *state, const BitSequence *data, DataLength databitlen);

void final_echo(sph_echo512_context *state, BitSequence *hashval);

void hash_echo(int hashbitlen, const BitSequence *data, DataLength databitlen, BitSequence *hashval);

void update_final_echo( sph_echo512_context *state, BitSequence *hashval,
                              const BitSequence *data, DataLength databitlen );
void echo_full( sph_echo512_context *state, BitSequence *hashval,
            int nHashSize, const BitSequence *data, DataLength databitlen );

void sph_echo512_init(void *cc);
void sph_echo512(void *cc, const void *data, size_t len);
void sph_echo512_close(void *cc, void *dst);

#ifdef __cplusplus
}
#endif


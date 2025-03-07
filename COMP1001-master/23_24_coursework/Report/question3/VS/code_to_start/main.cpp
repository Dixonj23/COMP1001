/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <dirent.h>

//function declarations
void Gaussian_Blur();
void Sobel();
int initialize_kernel();
void read_image(const char* filename);
void read_image_and_put_zeros_around(char* filename);
void write_image2(const char* filename, unsigned char* output_image);
void openfile(const char* filename, FILE** finput);
int getint(FILE* fp);

//CRITICAL POINT: images' paths - You need to change these paths
#define IN "C:\Users\mypc\OneDrive\Documents\GitHub\COMP1001\COMP1001-master\23_24_coursework\Report\question3\VS\code_to_start\input_images\a1.pgm"
#define OUT "C:\Users\mypc\OneDrive\Documents\GitHub\COMP1001\COMP1001-master\23_24_coursework\Report\question3\VS\code_to_start\output_images\blurred.pgm"
#define OUT2 "C:\Users\mypc\OneDrive\Documents\GitHub\COMP1001\COMP1001-master\23_24_coursework\Report\question3\VS\code_to_start\output_images\edge_detection.pgm"

//IMAGE DIMENSIONS
#define M 512  //cols
#define N 512  //rows


//CRITICAL POINT:these arrays are defined statically. Consider creating these arrays dynamically instead.
unsigned char frame1[N * M];//input image
unsigned char filt[N * M];//output filtered image
unsigned char gradient[N * M];//output image


const signed char Mask[5][5] = {//2d gaussian mask with integers
	{2,4,5,4,2} ,
	{4,9,12,9,4},
	{5,12,15,12,5},
	{4,9,12,9,4},
	{2,4,5,4,2}
};

const signed char GxMask[3][3] = {
	{-1,0,1} ,
	{-2,0,2},
	{-1,0,1}
};

const signed char GyMask[3][3] = {
	{-1,-2,-1} ,
	{0,0,0},
	{1,2,1}
};

char header[100];
errno_t err;

int main(int argc, char* argv[]) {
	if (argc != 4) {
		fprintf(stderr, "Usage: %s <input_image_path> <output_blurred_path> <output_edge_detection_path>\n", argv[0]);
		return 1;
	}

	//open the input_images folder

	DIR* dir;
	struct dirent* ent;
	if ((dir = opendir(argv[1])) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			if (ent->d_type == DT_REG) {
				char inputPath[255];
				char outputPath1[255];
				char outputPath2[255];

				sprintf(inputPath, "%s/%s", argv[1], ent->d_name);
				sprintf(outputPath1, "%s/blurred_%s", argv[2], ent->d_name);
				sprintf(outputPath2, "%s/edge_%s", argv[3], ent->d_name);

				read_image(inputPath);    // read image from disc

				Gaussian_Blur();        // blur the image (reduce noise)
				Sobel();                // apply edge detection

				write_image2(outputPath1, filt);    // store output image to the disc
				write_image2(outputPath2, gradient);    // store output image to the disc
			}
		}
		closedir(dir);
	}
	else {
		perror("Cannot open input folder");
		return EXIT_FAILURE;
	}
	
	return 0;
}




void Gaussian_Blur() {

	int row, col, rowOffset, colOffset;
	int newPixel;
	unsigned char pix;
	//const unsigned short int size=filter_size/2;
	const unsigned short int size = 2;

	/*---------------------- Gaussian Blur ---------------------------------*/
	for (row = 0; row < N; row++) {
		for (col = 0; col < M; col++) {
			newPixel = 0;
			for (rowOffset = -size; rowOffset <= size; rowOffset++) {
				for (colOffset = -size; colOffset <= size; colOffset++) {

					if ((row + rowOffset < 0) || (row + rowOffset >= N) || (col + colOffset < 0) || (col + colOffset >= M))
						pix = 0;
					else
						pix = frame1[M * (row + rowOffset) + col + colOffset];

					newPixel += pix * Mask[size + rowOffset][size + colOffset];

				}
			}
			filt[M * row + col] = (unsigned char)(newPixel / 159);

		}
	}

}


void Sobel() {
	int row, col, rowOffset, colOffset;
	__m128i Gx, Gy, pixel, mask;

	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col += 4) {

			Gx = _mm_setzero_si128();
			Gy = _mm_setzero_si128();

			for (rowOffset = -1; rowOffset <= 1; rowOffset++) {
				for (colOffset = -1; colOffset <= 1; colOffset++) {

					pixel = _mm_loadu_si123((__m128*) & filt[M * (row + rowOffset) + col + colOffset]);
					mask = _mm_set1_epi32(GxMask[rowOffset + 1][colOffset + 1]);

					Gx = _mm_add_epi32(Gx, _mm_mullo_epi32(pixel, mask));

					mask = _mm_set1_epi32(GyMask[rowOffset + 1][colOffset + 1])
					Gy = _mm_add_epi32(Gy, _mm_mullo_epi32(pixel, mask));
				}
			}
		}

		//gradient strength
		Gx = _mm_mullo_epi32(Gx, Gx);
		Gy = _mm_mullo_epi32(Gy, Gy);

		__m128i gradient_strength = _mm_add_epi32(Gx, Gy);
		gradient_strength =
			_mm_sqrt_epi32(gradient_strength);

		//store result

		_mm_storeu_si128((__m128i*) & gradient[M * row + col], _mm_packus_epi32(gradient_strength, _mm_setzero_si128()));
}



void read_image(const char* filename)
{

	int c;
	FILE* finput;
	int i, j, temp;

	printf("\nReading %s image from disk ...", filename);
	finput = NULL;
	openfile(filename, &finput);

	if ((header[0] == 'P') && (header[1] == '5')) { //if P5 image

		for (j = 0; j < N; j++) {
			for (i = 0; i < M; i++) {

				//if (fscanf_s(finput, "%d", &temp,20) == EOF)
				//	exit(EXIT_FAILURE);
				temp = getc(finput);

				frame1[M * j + i] = (unsigned char)temp;
			}
		}
	}
	else if ((header[0] == 'P') && (header[1] == '2'))  { //if P2 image
		for (j = 0; j < N; j++) {
			for (i = 0; i < M; i++) {

				if (fscanf_s(finput, "%d", &temp,20) == EOF)
					exit(EXIT_FAILURE);

				frame1[M * j + i] = (unsigned char)temp;
			}
		}
	}
	else {
		printf("\nproblem with reading the image");
		exit(EXIT_FAILURE);
	}

	fclose(finput);
	printf("\nimage successfully read from disc\n");

}



void write_image2(const char* filename, unsigned char* output_image)
{

	FILE* foutput;
	int i, j;



	printf("  Writing result to disk ...\n");

	if ((err = fopen_s(&foutput,filename, "wb")) != NULL) {
		fprintf(stderr, "Unable to open file %s for writing\n", filename);
		exit(-1);
	}

	fprintf(foutput, "P2\n");
	fprintf(foutput, "%d %d\n", M, N);
	fprintf(foutput, "%d\n", 255);

	for (j = 0; j < N; ++j) {
		for (i = 0; i < M; ++i) {
			fprintf(foutput, "%3d ", output_image[M * j + i]);
			if (i % 32 == 31) fprintf(foutput, "\n");
		}
		if (M % 32 != 0) fprintf(foutput, "\n");
	}
	fclose(foutput);


}




void openfile(const char* filename, FILE** finput)
{
	int x0, y0, x , aa;

	if (( err = fopen_s(finput,filename, "rb")) != NULL) {
		fprintf(stderr, "Unable to open file %s for reading\n", filename);
		exit(-1);
	}

	aa = fscanf_s(*finput, "%s", header, 20);

	x0 = getint(*finput);//this is M
	y0 = getint(*finput);//this is N
	printf("\t header is %s, while x=%d,y=%d", header, x0, y0);


	//CRITICAL POINT: AT THIS POINT YOU CAN ASSIGN x0,y0 to M,N 
	// printf("\n Image dimensions are M=%d,N=%d",M,N);


	x = getint(*finput); /* read and throw away the range info */
	//printf("\n range info is %d",x);

}



//CRITICAL POINT: you can define your routines here that create the arrays dynamically; now, the arrays are defined statically.



int getint(FILE* fp) /* adapted from "xv" source code */
{
	int c, i, firstchar;//, garbage;

	/* note:  if it sees a '#' character, all characters from there to end of
	   line are appended to the comment string */

	   /* skip forward to start of next number */
	c = getc(fp);
	while (1) {
		/* eat comments */
		if (c == '#') {
			/* if we're at a comment, read to end of line */
			char cmt[256], * sp;

			sp = cmt;  firstchar = 1;
			while (1) {
				c = getc(fp);
				if (firstchar && c == ' ') firstchar = 0;  /* lop off 1 sp after # */
				else {
					if (c == '\n' || c == EOF) break;
					if ((sp - cmt) < 250) *sp++ = c;
				}
			}
			*sp++ = '\n';
			*sp = '\0';
		}

		if (c == EOF) return 0;
		if (c >= '0' && c <= '9') break;   /* we've found what we were looking for */

		/* see if we are getting garbage (non-whitespace) */
	   // if (c!=' ' && c!='\t' && c!='\r' && c!='\n' && c!=',')
		//	garbage=1;

		c = getc(fp);
	}

	/* we're at the start of a number, continue until we hit a non-number */
	i = 0;
	while (1) {
		i = (i * 10) + (c - '0');
		c = getc(fp);
		if (c == EOF) return i;
		if (c < '0' || c>'9') break;
	}
	return i;
}









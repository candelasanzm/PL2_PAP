#pragma once
int ejecutarReduccion(float* dev_datos, int numVuelos, int esMax, int hilosPorBloque, double* tiempo_ms);
void tiempoKernels(float* dev_datos, int numVuelos, int* dev_resSimple, int* dev_resBasica, int* dev_resIntermedia, int esMax, int hilosPorBloque, dim3 dimGrid, dim3 dimBlock, double* tiempo_simple, double* tiempo_basica, double* tiempo_intermedia);
void ejecutarFase3(float* dep_delay, float* arr_delay, float* weather_delay, float* dep_time, float* arr_time, int numVuelos);
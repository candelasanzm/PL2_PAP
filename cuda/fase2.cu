#include <stdio.h> // para printf y scanf
#include <cuda_runtime.h> // para funciones Cuda
#include <math.h> // para que funcionen NAN o isnan
#include <stdlib.h> // para malloc, atoi, atof
#include <string.h> // para que strdup funcione

__constant__ int c_umbral; // declaramos la memoria constante

__global__ void fase2(float* dev_arrdelay, int numVuelos, int opcion, int* dev_contador, char* dev_tailNum, int* dev_indices, int* dev_tiempo) {
	int index = blockIdx.x * blockDim.x + threadIdx.x; // indice global

	if (index >= numVuelos) return; // corner check para evitar salirnos de los limites

	if (isnan(dev_arrdelay[index])) return; // descartamos las posiciones donde haya NAN para  evitar problemas

	int valor = (int)dev_arrdelay[index]; // definimos una variable donde guardamos el valor que haya en el array de datos en una posicion concreta
	char* matricula = dev_tailNum + index * 16; // accedemos a la matricula del vuelo actual en el array plano. Cada matricula ocupa 16 chars, por eso multiplicamos el indice por 16

	// Si la opcion elegida por el usuario es 1, es retrasos, por lo que el valor que se muestre por pantalla deberá ser mayor que el umbral dado por el usuario
	if ((opcion == 1) && (valor > c_umbral)) {
		printf("- Hilo %d | Matricula: %s | Retraso de %d minutos.\n", index, matricula, valor);
		int pos = atomicAdd(dev_contador, 1); // declaramos una variable posicion donde guardamos el contador, nos ayudara para hacer el array que se devuelve al host
		dev_tiempo[pos] = valor;
		dev_indices[pos] = index; // guardamos un array de indices que posteriormente se usara para recuperar las matriculas
	}

	// Si la opcion requerida es 2, adelantos, luego el valor sera menor que el umbral seleccionado, pero en negativo 
	else if ((opcion == 2) && (valor < -c_umbral)) {
		printf("- Hilo %d | Matricula: %s | Adelanto de %d minutos.\n", index, matricula, -valor);
		int pos = atomicAdd(dev_contador, 1);
		dev_tiempo[pos] = valor;
		dev_indices[pos] = index;
	}

	// Si la opcion seleccionada es 3, se buscan los despegues que se hayan hecho a tiempo, por lo que el valor de tiempo buscado es el 0
	else if ((opcion == 3) && (valor == 0)) {
		printf("- Hilo %d | Matricula: %s | Aterriza a tiempo.\n", index, matricula);
		int pos = atomicAdd(dev_contador, 1);
		dev_tiempo[pos] = valor;
		dev_indices[pos] = index;
	}
}

// Hacemos una función, donde ejecutaremos este kernel, que llamaremos desde el main
void ejecutarFase2(float* arr_delay, int numVuelos, char** tail_num) {

	// Pedimos al usuario la opcion que quiera realizar
	int opcion = -1;
	while (opcion != 0) {
		printf("\nElige como filtrar:\n");
		printf("1. Retrasos\n");
		printf("2. Adelantos\n");
		printf("3. Aterrizaje a tiempo\n");
		printf("4. Volver al Menu Principal\n");
		printf("Elija una opcion: ");

		scanf("%d", &opcion); // recogemos la opcion elegida por el usuario

		int umbral = -1; // variable donde recogeremos el umbral que quiera usar el usuario

		switch (opcion) {
		case 1: // haciendolo de esta manera el case 1 y el case 2 tienen el mismo codigo y usamos operadores ternarios en el kernel para simplificar
		case 2:
			while (umbral < 0) {
				printf("\nEscribe el umbral (en minutos) deseado: ");

				scanf("%d", &umbral); // leemos el umbral que el usuario quiera usar

				// Si el umbral es negativo dará un error y volvera a pedir un umbral valido
				if (umbral < 0) {
					printf("El umbral no puede ser negativo, introduce un valor positivo.\n");
				}
			}
		case 3: {
			// Copiar umbral en memoria constante
			cudaMemcpyToSymbol(c_umbral, &umbral, sizeof(int));

			// Comprobar caracteristicas hardware
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, 0);
			int hilosPorBloque = prop.maxThreadsPerBlock; // maximo de hilos por bloque
			int numBloques = (numVuelos + hilosPorBloque - 1) / hilosPorBloque; // calculamos el numero de bloques

			// CUDA no puede trabajar con un array de punteros a strings. Hay que crear un array plano para poder usar matriculas en la GPU
			char* h_tail = (char*)malloc(numVuelos * 16 * sizeof(char)); // reservamos espacio para almacenar todas las matriculas
			for (int i = 0; i < numVuelos; i++) {
				if (tail_num[i] != NULL) { // si hay matriculas la copiamos
					strncpy(h_tail + i * 16, tail_num[i], 15); // copiamos la matricula en su posicion del array plano, i*16 nos da la posicion de inicio de la matricula i en el array plano, 15 es el maximo de caracteres para dejar espacio al '\0' al final.
				}
				else {
					h_tail[i * 16] = '\0'; // si no hay matricula dejamos el espacio vacio
				}
			}

			// Reservamos espacion en memoria que ocuparemos con los arrays que devuelve la GPU a la CPU
			int* h_indices = (int*)malloc(numVuelos * sizeof(int));
			int* h_tiempo = (int*)malloc(numVuelos * sizeof(int));
			int h_contador = 0; // hacemos un contador para indicar cuantos vuelos salen

			// Declaramos los punteros
			int* dev_contador, * dev_tiempo, * dev_indices;
			float* dev_arrdelay;
			char* dev_tailNum;

			// Reservamos memoria en la GPU 
			cudaMalloc((void**)&dev_contador, sizeof(int));
			cudaMalloc((void**)&dev_arrdelay, numVuelos * sizeof(float));
			cudaMalloc((void**)&dev_tailNum, numVuelos * 16 * sizeof(char)); // Reservamos espacio de 16 chars de manera general
			cudaMalloc((void**)&dev_tiempo, numVuelos * sizeof(int));
			cudaMalloc((void**)&dev_indices, numVuelos * sizeof(int));

			// Copiamos los datos de la CPU en la GPU
			cudaMemcpy(dev_contador, &h_contador, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_arrdelay, arr_delay, numVuelos * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_tailNum, h_tail, numVuelos * 16 * sizeof(char), cudaMemcpyHostToDevice);

			// Ejecutamos el kernel adaptandonos a las caracteristicas del hardware de cada ordenador
			dim3 dimGrid(numBloques);
			dim3 dimBlock(hilosPorBloque);
			fase2 << <dimGrid, dimBlock >> > (dev_arrdelay, numVuelos, opcion, dev_contador, dev_tailNum, dev_indices, dev_tiempo);
			cudaDeviceSynchronize(); // esperamos a que todos los hilos terminen antes de continuar

			// Copiamos los datos de la GPU en la CPU
			cudaMemcpy(&h_contador, dev_contador, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_indices, dev_indices, h_contador * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_tiempo, dev_tiempo, h_contador * sizeof(int), cudaMemcpyDeviceToHost);

			// Escribimos por pantalla los vuelos totales que se han encontrado con las condiciones seleccionadas y el porcentaje que representan del total
			float porcentaje = (float)h_contador / (float)numVuelos * 100.0f; 
			if (h_contador == 0) {
				printf("No hay vuelos con el umbral solicitado\n");
			}
			else {
				printf("Se han encontrado %d vuelos con el umbral deseado. Porcentaje sobre el total: %.2f%%\n", h_contador, porcentaje);
				if (opcion == 1) {
					for (int i = 0; i < h_contador; i++) {
						printf("- Matricula %s. Retraso: %d minutos.\n", tail_num[h_indices[i]], h_tiempo[i]);
					}
				}
				else if (opcion == 2) {
					for (int i = 0; i < h_contador; i++) {
						printf("- Matricula %s. Adelanto: %d minutos.\n", tail_num[h_indices[i]], -h_tiempo[i]);
					}
				}
			}

			// Liberamos memoria
			cudaFree(dev_contador);
			cudaFree(dev_arrdelay);
			cudaFree(dev_tailNum);
			cudaFree(dev_indices);
			cudaFree(dev_tiempo);
			free(h_tail); // tambien hay que liberar los espacios del espacio que hemos guardado con malloc
			free(h_indices);
			free(h_tiempo);

			break; // para no entrar en el case 4
		}

		case 4: return; // el caso 4 es volver al menu principal

		// Si la opcion introducida no esta entre 1 y 3 mostramos un error
		default: 
			printf("Opcion no valida, introduzca un numero entre 1 y 4\n"); 
			break;
		}
	}
}
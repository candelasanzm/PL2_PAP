#include <stdio.h> // para printf y scanf
#include <cuda_runtime.h> // para funciones Cuda
#include <math.h> // para que funcionen NAN o isnan
#include <string.h>
#include "cloud.cuh"

// Definimos el kernel de lo que hara la fase 1
__global__ void fase1(float* dev_depdelay, int numVuelos, int umbral, int opcion, int* dev_contador) {
	int index = blockIdx.x * blockDim.x + threadIdx.x; // indice global

	if (index >= numVuelos) return; // corner check para evitar salirnos de los limites

	if (isnan(dev_depdelay[index])) return; // descartamos las posiciones donde haya NAN para  evitar problemas

	int valor = (int)dev_depdelay[index]; // definimos una variable donde guardamos el valor que haya en el array de datos en una posicion concreta
	
	// Si la opcion elegida por el usuario es 1, es retrasos, por lo que el valor que se muestre por pantalla deberá ser mayor que el umbral dado por el usuario
	if ((opcion == 1) && (valor > umbral)) {
		printf("Hilo %d. Retraso de %d minutos.\n", index, valor);
		atomicAdd(dev_contador, 1); // hacemos una operacion atomica donde vayamos actualizando el contador
	}

	// Si la opcion requerida es 2, adelantos, luego el valor sera menor que el umbral seleccionado, pero en negativo 
	else if ((opcion == 2) && (valor < -umbral)) {
		printf("Hilo %d. Adelanto de %d minutos.\n", index, -valor);
		atomicAdd(dev_contador, 1); // actualizamos el contador
	}

	// Si la opcion seleccionada es 3, se buscan los despegues que se hayan hecho a tiempo, por lo que el valor de tiempo buscado es el 0
	else if ((opcion == 3) && (valor == 0)) {
		printf("Hilo %d. Despegue a tiempo\n", index);
		atomicAdd(dev_contador, 1); // actualizamos el contador
	}
}

// Hacemos una función, donde ejecutaremos este kernel, que llamaremos desde el main
void ejecutarFase1(float* dep_delay, int numVuelos) {

	// Pedimos al usuario la opcion que quiera realizar
	int opcion = -1;
	while (opcion != 0) {
		printf("\nElige como filtrar: \n");
		printf("1. Retrasos\n");
		printf("2. Adelantos\n");
		printf("3. Despegues a tiempo\n");
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
				// Comprobar caracteristicas hardware
				cudaDeviceProp prop;
				cudaGetDeviceProperties(&prop, 0);
				int hilosPorBloque = prop.maxThreadsPerBlock; // maximo de hilos por bloque
				int numBloques = (numVuelos + hilosPorBloque - 1) / hilosPorBloque; // calculamos el numero de bloques

				int h_contador = 0; // hacemos un contador para indicar cuantos vuelos salen

				// Declaramos los punteros
				int* dev_contador;
				float* dev_depdelay;

				// Reservamos memoria en la GPU
				cudaMalloc((void**)&dev_contador, sizeof(int));
				cudaMalloc((void**)&dev_depdelay, numVuelos * sizeof(float));

				// Copiamos los datos de la CPU en la GPU
				cudaMemcpy(dev_contador, &h_contador, sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_depdelay, dep_delay, numVuelos * sizeof(float), cudaMemcpyHostToDevice);

				// Ejecutamos el kernel adaptandonos a las caracteristicas del hardware de cada ordenador
				dim3 dimGrid(numBloques);
				dim3 dimBlock(hilosPorBloque);
				fase1 << <dimGrid, dimBlock >> > (dev_depdelay, numVuelos, umbral, opcion, dev_contador);
				cudaDeviceSynchronize(); // esperamos a que todos los hilos terminen antes de continuar

				// Copiamos los datos de la GPU en la CPU
				cudaMemcpy(&h_contador, dev_contador, sizeof(int), cudaMemcpyDeviceToHost);

				// Escribimos por pantalla los vuelos totales que se han encontrado con las condiciones seleccionadas y el porcentaje que representan del total
				float porcentaje = (float)h_contador / numVuelos * 100.0f;
				if (h_contador == 0) {
					printf("No hay vuelos con el umbral solicitado\n");
				}
				else {
					printf("Se han encontrado %d vuelos con el umbral deseado. Porcentaje sobre el total: %.2f%%\n", h_contador, porcentaje);
				}

				// Liberamos memoria
				cudaFree(dev_contador);
				cudaFree(dev_depdelay);

				// ============================================================
				// CLOUD: preguntar si subir resultados
				// ============================================================
				char nombreFase[100];
				char opcionesEntrada[256];
				char resultados[10][256];
				int numResultados = 0;

				if (opcion == 1) {
					snprintf(nombreFase, sizeof(nombreFase), "Fase 1 - Retrasos en Despegues");
					snprintf(opcionesEntrada, sizeof(opcionesEntrada), "umbral = %d minutos, total_vuelos = %d", umbral, numVuelos);
				} else if (opcion == 2) {
					snprintf(nombreFase, sizeof(nombreFase), "Fase 1 - Adelantos en Despegues");
					snprintf(opcionesEntrada, sizeof(opcionesEntrada), "umbral = %d minutos, total_vuelos = %d", umbral, numVuelos);
				} else if (opcion == 3) {
					snprintf(nombreFase, sizeof(nombreFase), "Fase 1 - Despegues a tiempo");
					snprintf(opcionesEntrada, sizeof(opcionesEntrada), "total_vuelos = %d", numVuelos);
				}

				snprintf(resultados[0], sizeof(resultados[0]), "Vuelos encontrados: %d (%.2f%% del total)", h_contador, porcentaje);

				preguntar_y_enviar(nombreFase, opcionesEntrada, resultados, numResultados);

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
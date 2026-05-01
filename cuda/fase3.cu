#include <stdio.h> // para printf y scanf
#include <cuda_runtime.h> // para funciones Cuda
#include <math.h> // para que funcionen NAN o isnan
#include <limits.h> // para INT_MIN e INT_MAX
#include <time.h> // para poder medir el tiempo
#include "cloud.cuh" // para poder llamar a funciones del fichero cloud

// Definimos un kernel para realizar los calculos de la parte simple
__global__ void fase3_simple(float* dev_datos, int numVuelos, int* dev_resSimple, int esMax) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; // indice global

	if (i >= numVuelos) return; // corner check para evitar salirnos de los limites

	if (isnan(dev_datos[i])) return; // descartamos las posiciones donde haya NAN para  evitar problemas

	int valor = (int)dev_datos[i]; // definimos una variable donde guardamos el valor que haya en el array de datos en una posicion concreta
	
	// En funcion de si el usuario pide que se calcule el maximo o el minimo hacemos una operacion u otra
	if (esMax) {
		atomicMax(dev_resSimple, valor);
	}
	else {
		atomicMin(dev_resSimple, valor);
	}
}

// Definimos un kernel para realizar los calculos de la parte basica
__global__ void fase3_basica(float* dev_datos, int numVuelos, int* dev_resBasica, int esMax) {
	__shared__ float datos_memoria[1024]; // definimos la memoria compartida

	int i = blockIdx.x * blockDim.x + threadIdx.x; // indice global
	
	datos_memoria[threadIdx.x] = NAN; // Inicializamos la memoria compartida a NAN para que los hilos que no tengan datos validos no interfieran en los calculos
	__syncthreads(); // esperamos a que todos los hilos terminen antes de continuar

	// Trabajamos con los datos sin salirnos de los limites y excluyendo los nan para evitar problemas
	if (i < numVuelos && !isnan(dev_datos[i])) {
		datos_memoria[threadIdx.x] = dev_datos[i]; // guardamos en cada posicion de memoria su posicion en el array que introducimos
	}
	
	__syncthreads(); // esperamos a que todos los hilos terminen antes de continuar

	// Solo procesamos hilos validos con datos no NaN
	if (i >= numVuelos || isnan(dev_datos[i])) return;

	int posActual = (int)datos_memoria[threadIdx.x]; // la posicion actual es la posicion de memoria del hilo
	
	// Para el anterior: comprobar que no sea el primer hilo del bloque y que no sea NAN
	int posAnterior = posActual;
	if (threadIdx.x > 0 && !isnan(datos_memoria[threadIdx.x - 1])) {
		posAnterior = (int)datos_memoria[threadIdx.x - 1];
	}

	// Para el posterior comprobar que no sea el ultimo hilo del bloque y que no sea NAN
	int posPosterior = posActual;
	if (threadIdx.x + 1 < blockDim.x && i + 1 < numVuelos && !isnan(datos_memoria[threadIdx.x + 1])) {
		posPosterior = (int)datos_memoria[threadIdx.x + 1];
	}

	// declaramos una variable resultado donde guardaremos el maximo de las tres posiciones contiguas
	int resultado;

	// En funcion de si el usuario pide que se calcule el maximo o el minimo hacemos una operacion u otra
	if (esMax) {
		resultado = max(max(posAnterior, posActual), posPosterior);
		atomicMax(dev_resBasica, resultado);
	}
	else {
		resultado = min(min(posAnterior, posActual), posPosterior);
		atomicMin(dev_resBasica, resultado);
	}
}

// Definimos un kernel para realizar los calculos de la parte intermedia
__global__ void fase3_intermedia(float* dev_datos, int numVuelos, int* dev_resIntermedia, int esMax) {
	__shared__ float datos_memoria[1024]; // definimos la memoria compartida

	int i = blockIdx.x * blockDim.x + threadIdx.x; // indice global
	 
	datos_memoria[threadIdx.x] = NAN;  // Inicializamos la memoria compartida a NAN para que los hilos que no tengan datos validos no interfieran en los calculos
	__syncthreads(); // esperamos a que todos los hilos terminen antes de continua

	// Trabajamos con los datos sin salirnos de los limites y excluyendo los nan para evitar problemas
	if (i < numVuelos && !isnan(dev_datos[i])) {
		datos_memoria[threadIdx.x] = dev_datos[i]; // guardamos en cada posicion de memoria su posicion en el array que introducimos
	}

	__syncthreads(); // esperamos a que todos los hilos terminen antes de continuar

	// Solo procesamos hilos validos con datos no NaN y que no se salgan de los limites
	if (i >= numVuelos || isnan(dev_datos[i])) return;

	int posActual = (int)datos_memoria[threadIdx.x]; // la posicion actual es la posicion de memoria del hilo

	// Para el anterior: comprobar que no sea el primer hilo del bloque y que no sea NAN
	int posAnterior = posActual;
	if (threadIdx.x > 0 && !isnan(datos_memoria[threadIdx.x - 1])) {
		posAnterior = (int)datos_memoria[threadIdx.x - 1];
	}

	// Para el posterior comprobar que no sea el ultimo hilo del bloque y que no sea NAN
	int posPosterior = posActual;
	if (threadIdx.x + 1 < blockDim.x && i + 1 < numVuelos && !isnan(datos_memoria[threadIdx.x + 1])) {
		posPosterior = (int)datos_memoria[threadIdx.x + 1];
	}

	// Guardamos el maximo y minimo en memoria compartida
	int resultado;
	if (esMax) {
		resultado = max(max(posAnterior, posActual), posPosterior);
	}
	else {
		resultado = min(min(posAnterior, posActual), posPosterior);
	}

	datos_memoria[threadIdx.x] = resultado;
	__syncthreads(); // esperamos a que todos los hilos terminen antes de continuar

	// Solo trabajamos con los hilos cuyo identificador sea par
	if (threadIdx.x % 2 == 0) {
		int actual = (int)datos_memoria[threadIdx.x]; // la posicion actual es la posicion de memoria del hilo

		// Para el posterior comprobar que no sea el ultimo hilo del bloque y que no sea NAN
		int siguiente = actual;
		if (threadIdx.x + 1 < blockDim.x && i + 1 < numVuelos && !isnan(datos_memoria[threadIdx.x + 1])) {
			siguiente = (int)datos_memoria[threadIdx.x + 1];
		}

		// Realizamos la comparacion pedida y la guardamos en la variable resultado 
		int comparacion;
		if (esMax) {
			comparacion = max(actual, siguiente);
			atomicMax(dev_resIntermedia, comparacion);
		}
		else {
			comparacion = min(actual, siguiente);
			atomicMin(dev_resIntermedia, comparacion);
		}
	}
}

// Definimos un kernel para realizar los calculos de la parte de reduccion
__global__ void fase3_reduccion(float* dev_datos, int numVuelos, int* dev_resReduccion, int esMax) {
	__shared__ int datos_memoria[1024]; // definimos la memoria compartida

	int i = blockIdx.x * blockDim.x + threadIdx.x; // indice global

	// Trabajamos con los datos sin salirnos de los limites y excluyendo los nan para evitar problemas
	if (i < numVuelos && !isnan(dev_datos[i])) {
		datos_memoria[threadIdx.x] = (int)dev_datos[i]; // guardamos en cada posicion de memoria su posicion en el array que introducimos
	}
	else {
		// Si el hilo no tiene dato valido lo inicializamos con el valor limite para que no interfiera en la comparacion
		if (esMax) {
			datos_memoria[threadIdx.x] = INT_MIN; // para maximo (cualquier valor real sera mayor)
		}
		else {
			datos_memoria[threadIdx.x] = INT_MAX; // para minimo(cualquier valor real sera menor)
		}
	}

	__syncthreads(); // esperamos a que todos los hilos terminen antes de continuar

	// Reducción en árbol dentro del bloque: en cada iteracion la mitad de los hilos activos compara su valor con el hilo que esta a stride posiciones
	// El stride se divide a la mitad en cada iteracion hasta que el hilo 0 tiene el resultado
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			if (esMax) {
				if (datos_memoria[threadIdx.x + stride] > datos_memoria[threadIdx.x]) {
					datos_memoria[threadIdx.x] = datos_memoria[threadIdx.x + stride];
				}
			}
			else {
				if (datos_memoria[threadIdx.x + stride] < datos_memoria[threadIdx.x]) {
					datos_memoria[threadIdx.x] = datos_memoria[threadIdx.x + stride];
				}
			}
		}

		__syncthreads(); // esperamos a que todos los hilos terminen antes de continuar
	}

	// El hilo 0 escribe el resultado
	if (threadIdx.x == 0) {
		dev_resReduccion[blockIdx.x] = datos_memoria[0]; // El hilo 0 de cada bloque escribe el resultado parcial de su bloque en su posicion del array de memoria
	}
}

// Ahora hacemos una función para hacerla recursiva
__global__ void fase3_reduccion_int(int* dev_datos, int numVuelos, int* dev_resReduccion, int esMax) {
	__shared__ int datos_memoria[1024]; // declaramos la memoria compartida

	int i = blockIdx.x * blockDim.x + threadIdx.x; // indice global

	// Trabajamos con los datos sin salirnos de los limites y excluyendo los nan para evitar problemas
	if (i < numVuelos) {
		datos_memoria[threadIdx.x] = dev_datos[i]; // guardamos en cada posicion de memoria su posicion en el array que introducimos
	}
	else {
		// Usamos unos limites para evitar comparar con datos basura
		if (esMax) {
			datos_memoria[threadIdx.x] = INT_MIN;
		}
		else {
			datos_memoria[threadIdx.x] = INT_MAX;
		}
	}

	__syncthreads(); // esperamos a que todos los hilos terminen antes de continuar

	// Reducción en árbol dentro del bloque: en cada iteracion la mitad de los hilos activos compara su valor con el hilo que esta a stride posiciones
	// El stride se divide a la mitad en cada iteracion hasta que el hilo 0 tiene el resultado
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			if (esMax) {
				if (datos_memoria[threadIdx.x + stride] > datos_memoria[threadIdx.x]) {
					datos_memoria[threadIdx.x] = datos_memoria[threadIdx.x + stride];
				}
			}
			else {
				if (datos_memoria[threadIdx.x + stride] < datos_memoria[threadIdx.x]) {
					datos_memoria[threadIdx.x] = datos_memoria[threadIdx.x + stride];
				}
			}
		}

		__syncthreads(); // esperamos a que todos los hilos terminen antes de continuar
	}

	// El hilo 0 escribe el resultado
	if (threadIdx.x == 0) {
		dev_resReduccion[blockIdx.x] = datos_memoria[0]; // El hilo 0 de cada bloque escribe el resultado parcial de su bloque en su posicion del array de memoria
	}
}

// Funcion para llamar desde el host
int ejecutarReduccion(float* dev_datos, int numVuelos, int esMax, int hilosPorBloque, double* tiempo_ms) {
	clock_t inicio = clock();

	// Llamamos a la primera funcion, con los datos iniciales del programa
	int n = (numVuelos + hilosPorBloque - 1) / hilosPorBloque; // Calculamos el numero de bloques necesarios, que sera el tamaño del array de resultados parciales

	int* dev_temp; // definimos un puntero que usaremos como temporal para guardar los resultados de la primera funcion que es la que lo hace con float y solo se usa una vez

	cudaMalloc((void**)&dev_temp, n * sizeof(int)); // reservamos espacio de memoria para el temporal

	fase3_reduccion << <n, hilosPorBloque >> > (dev_datos, numVuelos, dev_temp, esMax); // ejecutamos la primera funcion, la que convierte los datos float de entrada en los int que usaremos despues
	cudaDeviceSynchronize(); // esperamos a que todos los hilos terminen antes de continuar

	// Llamamos a la segunda funcion recursivamente hasta tener 10 elementos
	while (n > 10) {
		int bloques = (n + hilosPorBloque - 1) / hilosPorBloque; // calculamos el numero de bloques que usaremos en el lanzamiento del kernel
		int* dev_reduccion; // definimos el puntero que usaremos para guardar la salida
		cudaMalloc((void**)&dev_reduccion, bloques * sizeof(int)); // reservamos el espacio en memoria
		fase3_reduccion_int << <bloques, hilosPorBloque >> > (dev_temp, n, dev_reduccion, esMax); // ejecutamos el kernel adaptandonos a las características del hardware
		cudaDeviceSynchronize(); // esperamos a que todos los hilos terminen antes de continuar
		cudaFree(dev_temp); // liberamos la memoria
		dev_temp = dev_reduccion; // en el temporal dejamos solo los valores que tenemos en la salida para seguir reduciendo
		n = bloques; // la n ahora sera el numero de bloques para seguir con la recursividad
	}

	// Copiamos los 10 valores en el host
	int* h_temp = (int*)malloc(n * sizeof(int)); // reservamos el espacio de memoria
	cudaMemcpy(h_temp, dev_temp, n * sizeof(int), cudaMemcpyDeviceToHost); // copiamos los datos de la GPU en la CPU
	cudaFree(dev_temp); // liberamos el espacio de memoria usado por el temporal

	// Iteramos el vector final de maximo 10 elementos en la CPU para obtener el resultado definitivo
	int resultado = h_temp[0];
	for (int i = 1; i < n; i++) {
		if (esMax) {
			if (h_temp[i] > resultado) {
				resultado = h_temp[i];
			} 
		}
		else {
			if (h_temp[i] < resultado) {
				resultado = h_temp[i];
			}
		}
	}
	
	free(h_temp); // liberamos la memoria

	clock_t fin = clock();
	*tiempo_ms = (double)(fin - inicio) / CLOCKS_PER_SEC * 1000.0;

	return resultado; // devolvemos el resulrado
}

// Hacemos una funcion que nos permita medir el tiempo que tarda cada metodo en ejecutarse para poder compararlos 
void tiempoKernels(float* dev_datos, int numVuelos, int* dev_resSimple, int* dev_resBasica, int* dev_resIntermedia, int esMax, int hilosPorBloque, dim3 dimGrid, dim3 dimBlock, double* tiempo_simple, double* tiempo_basica, double* tiempo_intermedia) {
	clock_t inicio, fin;

	inicio = clock(); // empezamos el tiempo
	fase3_simple << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resSimple, esMax);
	cudaDeviceSynchronize(); // esperamos a que todos los hilos terminen antes de continuar
	fin = clock(); // acabamos el tiempo
	*tiempo_simple = (double)(fin - inicio) / CLOCKS_PER_SEC * 1000.0; // contamos cuanto tiempo ha tardado la fase simple en ejecutarse

	inicio = clock(); // empezamos el tiempo
	fase3_basica << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resBasica, esMax);
	cudaDeviceSynchronize(); // esperamos a que todos los hilos terminen antes de continuar
	fin = clock(); // acabamos el tiempo
	*tiempo_basica = (double)(fin - inicio) / CLOCKS_PER_SEC * 1000.0; // contamos cuanto tiempo ha tardado la fase basica en ejecutarse

	inicio = clock(); // empezamos el tiempo
	fase3_intermedia << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resIntermedia, esMax);
	cudaDeviceSynchronize(); // esperamos a que todos los hilos terminen antes de continuar
	fin = clock(); // acabamos el tiempo
	*tiempo_intermedia = (double)(fin - inicio) / CLOCKS_PER_SEC * 1000.0; // contamos cuanto tiempo ha tardado la fase intermedia en ejecutarse
}

// Funcion auxiliar que construye los datos para el cloud y llama a la funcion donde preguntar si enviamos al cloud. Hacemos una funcion par abreviar el codigo
void enviarResultadosFase3(const char* campo, int esMax, int res_simple, double tiempo_simple, int res_basica, double tiempo_basica, int res_intermedia, double tiempo_intermedia, int res_reduccion, double tiempo_reduccion, int numVuelos) {
	char nombreFase[100]; // buffer donde guardamos el nombre que identifica la fase
	char opcionesEntrada[256]; // buffer donde guardamos los parametros de entrada
	char resultados[10][256]; // array de hasta 10 resultados cada uno de hasta 256 caracteres

	// Construimos el nombre de la fase y la operacion
	snprintf(nombreFase, sizeof(nombreFase), "Fase 3 - %s (%s)", campo, esMax ? "Maximo" : "Minimo");
	snprintf(opcionesEntrada, sizeof(opcionesEntrada), "Campo = %s, Operacion = %s, Total vuelos = %d", campo, esMax ? "max" : "min", numVuelos);
	snprintf(resultados[0], sizeof(resultados[0]), "Simple: valor = %d, tiempo = %.3f ms", res_simple, tiempo_simple);
	snprintf(resultados[1], sizeof(resultados[1]), "Basica: valor = %d, tiempo = %.3f ms", res_basica, tiempo_basica);
	snprintf(resultados[2], sizeof(resultados[2]), "Intermedia: valor = %d, tiempo = %.3f ms", res_intermedia, tiempo_intermedia);
	snprintf(resultados[3], sizeof(resultados[3]), "Reduccion: valor = %d, tiempo = %.3f ms", res_reduccion, tiempo_reduccion);

	preguntar_y_enviar(nombreFase, opcionesEntrada, resultados, 4);
}

// Funcion que ejecutaremos en el programa principal para usar la fase 3
void ejecutarFase3(float* dep_delay, float* arr_delay, float* weather_delay, float* dep_time, float* arr_time, int numVuelos) {

	// Pedimos la opcion
	int opcion = -1;
	while (opcion != 0) {
		printf("\nElige como filtrar: \n");
		printf("1. Retraso en Despegue\n");
		printf("2. Retraso en Aterrizaje\n");
		printf("3. Retraso por Condiciones Meteorologicas\n");
		printf("4. Hora de Despegue\n");
		printf("5. Hora de Aterrizaje\n");
		printf("6. Volver al Menu Principal\n");
		printf("Elija una opcion: ");

		// Validamos la respuesta del usuario
		if (scanf("%d", &opcion) != 1) {
			printf("Entrada no valida.\n");
			int c;
			while ((c = getchar()) != '\n' && c != EOF);
			continue;
		}

		switch (opcion) {
			case 1:
			case 2:
			case 3:
			case 4:
			case 5: {
				// Pedimos una nueva opcion
				int volver = 0; // para volver al menu de opcion
				int opcion2 = -1;
				while (opcion2 != 0) {
					printf("\nOpciones Disponibles\n");
					printf("1. Maximo\n");
					printf("2. Minimo\n");
					printf("3. Volver Atras\n");
					printf("Seleccione operacion: ");

					if (scanf("%d", &opcion2) != 1) {
						printf("Entrada no valida.\n");
						int c;
						while ((c = getchar()) != '\n' && c != EOF);
						continue;
					}

					switch (opcion2) {
					case 1:
					case 2: {
						// Comprobar caracteristicas hardware
						cudaDeviceProp prop;
						cudaGetDeviceProperties(&prop, 0);
						int hilosPorBloque = prop.maxThreadsPerBlock; // maximo de hilos por bloque
						int numBloques = (numVuelos + hilosPorBloque - 1) / hilosPorBloque; // calculamos el numero de bloques que usaremos

						// Inicializamos los resultados segun la operacion:
						// Para maximo usamos INT_MIN para que cualquier valor sea mayor
						// Para minimo usamos INT_MAX para que cualquier valor sea menor
						int valorInicial = (opcion2 == 1) ? INT_MIN : INT_MAX; // si la opcion2 es 1 valorInicial sera INT_MIN, en caso contrario sera INT_MAX
						int res_simple = valorInicial; 
						int res_basica = valorInicial;
						int res_intermedia = valorInicial;
						int res_reduccion = valorInicial;

						// declaramos los punteros
						float* dev_depDelay, * dev_arrDelay, * dev_weatherDelay, * dev_depTime, * dev_arrTime;
						int* dev_resSimple, * dev_resBasica, * dev_resIntermedia;

						// Reservamos el espacio en memoria
						cudaMalloc((void**)&dev_depDelay, numVuelos * sizeof(float));
						cudaMalloc((void**)&dev_arrDelay, numVuelos * sizeof(float));
						cudaMalloc((void**)&dev_weatherDelay, numVuelos * sizeof(float));
						cudaMalloc((void**)&dev_depTime, numVuelos * sizeof(float));
						cudaMalloc((void**)&dev_arrTime, numVuelos * sizeof(float));
						cudaMalloc((void**)&dev_resSimple, sizeof(int));
						cudaMalloc((void**)&dev_resBasica, sizeof(int));
						cudaMalloc((void**)&dev_resIntermedia, sizeof(int));

						// Transferimos información de la CPU a la GPU
						cudaMemcpy(dev_depDelay, dep_delay, numVuelos * sizeof(float), cudaMemcpyHostToDevice);
						cudaMemcpy(dev_arrDelay, arr_delay, numVuelos * sizeof(float), cudaMemcpyHostToDevice);
						cudaMemcpy(dev_weatherDelay, weather_delay, numVuelos * sizeof(float), cudaMemcpyHostToDevice);
						cudaMemcpy(dev_depTime, dep_time, numVuelos * sizeof(float), cudaMemcpyHostToDevice);
						cudaMemcpy(dev_arrTime, arr_time, numVuelos * sizeof(float), cudaMemcpyHostToDevice);
						cudaMemcpy(dev_resSimple, &valorInicial, sizeof(int), cudaMemcpyHostToDevice);
						cudaMemcpy(dev_resBasica, &valorInicial, sizeof(int), cudaMemcpyHostToDevice);
						cudaMemcpy(dev_resIntermedia, &valorInicial, sizeof(int), cudaMemcpyHostToDevice);

						dim3 dimGrid(numBloques);
						dim3 dimBlock(hilosPorBloque);

						const char* nombreCampo = ""; // string que identifica el campo para mostrar y enviar al cloud
						float* dev_datos = NULL; // puntero al array de datos en GPU correspondiente al campo elegido

						// Dependiendo de la opcion seleccionada por el usuario dev_datos y nombreCampo sera un campo distinto
						if (opcion == 1) {
							dev_datos = dev_depDelay;
							nombreCampo = "DEP_DELAY";
						}
						else if (opcion == 2) {
							dev_datos = dev_arrDelay;
							nombreCampo = "ARR_DELAY";
						}
						else if (opcion == 3) {
							dev_datos = dev_weatherDelay;
							nombreCampo = "WEATHER_DELAY";
						}
						else if (opcion == 4) {
							dev_datos = dev_depTime;
							nombreCampo = "DEP_TIME";
						}
						else if (opcion == 5) {
							dev_datos = dev_arrTime;
							nombreCampo = "ARR_TIME";
						}

						// Dependiendo lo que el usuario elija vemos si quiere maximo (1) o minimo (0)
						int esMax = (opcion2 == 1) ? 1 : 0;
						
						// Ejecutamos los kernel
						double tiempo_simple, tiempo_basica, tiempo_intermedia, tiempo_reduccion; // declaramos unas variables donde mostraremos el tiempo que ha tardado cada metodo
						tiempoKernels(dev_datos, numVuelos, dev_resSimple, dev_resBasica, dev_resIntermedia, esMax, hilosPorBloque, dimGrid, dimBlock, &tiempo_simple, &tiempo_basica, &tiempo_intermedia);
						res_reduccion = ejecutarReduccion(dev_datos, numVuelos, esMax, hilosPorBloque, &tiempo_reduccion);

						// Transferimos la informacion de la GPU a la CPU
						cudaMemcpy(&res_simple, dev_resSimple, sizeof(int), cudaMemcpyDeviceToHost);
						cudaMemcpy(&res_basica, dev_resBasica, sizeof(int), cudaMemcpyDeviceToHost);
						cudaMemcpy(&res_intermedia, dev_resIntermedia, sizeof(int), cudaMemcpyDeviceToHost);

						// Imprimimos los resultados en formato tabla donde podemos observar el metodo usado, el resultado obtenido, y el tiempo tardado en cada una de las distintas posibilidades (simple, basica...)
						printf("\n+-----------------+------------+------------+\n");
						printf("| %-15s | %10s | %10s |\n", nombreCampo, "Resultado", "Tiempo ms");
						printf("+-----------------+------------+------------+\n");
						printf("| %-15s | %10d | %10.3f |\n", "Simple", res_simple, tiempo_simple);
						printf("| %-15s | %10d | %10.3f |\n", "Basica", res_basica, tiempo_basica);
						printf("| %-15s | %10d | %10.3f |\n", "Intermedia", res_intermedia, tiempo_intermedia);
						printf("| %-15s | %10d | %10.3f |\n", "Reduccion", res_reduccion, tiempo_reduccion);
						printf("+-----------------+------------+------------+\n");
						
						// ============================================================
						// CLOUD: enviar los datos
						// ============================================================
						enviarResultadosFase3(nombreCampo, esMax, res_simple, tiempo_simple, res_basica, tiempo_basica, res_intermedia, tiempo_intermedia, res_reduccion, tiempo_reduccion, numVuelos);

						// Liberamos la memoria
						cudaFree(dev_depDelay);
						cudaFree(dev_arrDelay);
						cudaFree(dev_weatherDelay);
						cudaFree(dev_depTime);
						cudaFree(dev_arrTime);
						cudaFree(dev_resSimple);
						cudaFree(dev_resBasica);
						cudaFree(dev_resIntermedia);

						break; // salimos 
					}

					case 3:
						volver = 1; // volvemos a elegir
						break; 

					// En caso de que la opcion no este entre 1 y 3 mostramos un error y pedimos un nuevo valor
					default:
						printf("Opcion no valida, introduzca un numero entre 1 y 3\n");
						break;
					}

					if (volver) break;
				}
				break;
			}

			case 6: return;

			// En caso de que la opcion no este entre 1 y 6 mostramos un error y pedimos un nuevo valor
			default:
				printf("Opcion no valida, introduzca un numero entre 1 y 6\n");
				break;
		}
	}
}
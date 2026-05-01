#include <stdio.h> // para printf y scanf
#include <cuda_runtime.h> // para funciones Cuda
#include <string.h> // para que memset funcione
#include <stdlib.h> // para malloc
#include "cloud.cuh" // para poder llamar a funciones del fichero cloud

// Definimos el kernel que lo que hara es contar el numero de vuelos por aeropuerto
__global__ void fase4_datos(int* dev_datos, int numVuelos, int* dev_contador) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numVuelos) return; // nos aseguramos de estar en los limites del dataset

	int valor = dev_datos[index];
	atomicAdd(&dev_contador[valor], 1); // sumamos uno al contador de vuelos
}

void ejecutarFase4(int* origin_seq_id, int* dest_seq_id, char** origin_airport, char** dest_airport, int numVuelos) {
	// Solicitamos al usuario el numero que nos indique que quiere hacer
	int opcion = -1;
	while (opcion != 0) {
		printf("\nTipo de Aeropuerto:\n");
		printf("1. Aeropuerto de Origen\n");
		printf("2. Aeropuerto de Destino\n");
		printf("3. Buscar un Aeropuerto por su Codigo\n");
		printf("4. Top 5 Aeropuertos Mas Concurridos\n");
		printf("5. Top 5 Aeropuertos Menos Concurridos\n");
		printf("6. Volver Atras\n");
		printf("Seleccione una opcion: ");

		// Validamos la respuesta del usuario
		if (scanf("%d", &opcion) != 1) {
			printf("Entrada no valida.\n");
			int c;
			while ((c = getchar()) != '\n' && c != EOF);
			continue;
		}

		int umbral = -1; // variable donde recogeremos el umbral que quiera usar el usuario

		switch (opcion) {
			case 1:
			case 2: {
				while (umbral < 0) {
					printf("\nEscribe el umbral (en minutos) deseado: ");

					if (scanf("%d", &umbral) != 1) {
						printf("Entrada no valida.\n");
						int c;
						while ((c = getchar()) != '\n' && c != EOF);
						continue;
					}

					// Si el umbral es negativo dará un error y volvera a pedir un umbral valido
					if (umbral < 0) {
						printf("El umbral no puede ser negativo, introduce un valor positivo.\n");
					}
				}

				// Comprobar caracteristicas hardware
				cudaDeviceProp prop;
				cudaGetDeviceProperties(&prop, 0);
				int hilosPorBloque = prop.maxThreadsPerBlock; // maximo de hilos por bloque
				int numBloques = (numVuelos + hilosPorBloque - 1) / hilosPorBloque; // calculamos el numero de bloques

				// Definimos los punteros (haciendo uso de ternarios para simplificar el codigo) 
				int* seq_id = (opcion == 1) ? origin_seq_id : dest_seq_id; // si la opcion es 1 trabajamos con aeropuertos de origen, si es 2 trabajamos con los de destino
				char** aeropuerto = (opcion == 1) ? origin_airport : dest_airport;
				const char* tipo = (opcion == 1) ? "origen" : "destino";

				// Necesitamos saber cual es el ID maximo para reservar espacio en memoria
				int maxID = 0;
				for (int i = 0; i < numVuelos; i++) {
					if (seq_id[i] > maxID) {
						maxID = seq_id[i];
					}
				}

				// Definimos el contador que en este caso sera un array
				int* h_contadores = (int*)malloc((maxID + 1) * sizeof(int));
				if (h_contadores == NULL) {
					printf("ERROR: No se pudo reservar memoria CPU.\n");
					break;
				}
				memset(h_contadores, 0, (maxID + 1) * sizeof(int)); // inicializamos el array con todo ceros

				int* dev_datos, * dev_contador; // definimos los punteros

				// Reservamos el espacio en memoria
				cudaMalloc((void**)&dev_datos, numVuelos * sizeof(int));
				cudaMalloc((void**)&dev_contador, (maxID + 1) * sizeof(int));

				// Transferimos los datos de la CPU a la GPU 
				cudaMemcpy(dev_datos, seq_id, numVuelos * sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_contador, h_contadores, (maxID + 1) * sizeof(int), cudaMemcpyHostToDevice);

				// Ejecutamos el kernel adaptandonos a las caracteristicas del hardware de cada ordenador
				dim3 dimGrid(numBloques);
				dim3 dimBlock(hilosPorBloque);
				fase4_datos << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_contador);
				cudaDeviceSynchronize(); // esperamos a que todos los hilos terminen antes de continuar

				// Transferimos los datos de la GPU a la CPU 
				cudaMemcpy(h_contadores, dev_contador, (maxID + 1) * sizeof(int), cudaMemcpyDeviceToHost);

				// Calculamos cuantos aeropuertos que cumplan la condicion hay
				int aeropuertosUnicos = 0;
				for (int i = 0; i <= maxID; i++) {
					if (h_contadores[i] > 0) {
						aeropuertosUnicos++;
					}
				}
				printf("\nNumero de aeropuertos de %s encontrados: %d\n", tipo, aeropuertosUnicos);

				// Relacionamos los ids de los aeropuertos con su codigo
				char** mapaAeropuertos = (char**)malloc((maxID + 1) * sizeof(char*)); // reservamos el espacio en memoria 
				memset(mapaAeropuertos, 0, (maxID + 1) * sizeof(char*)); // inicializamos todo el array a 0 
				for (int i = 0; i < numVuelos; i++) {
					if (seq_id[i] > 0 && aeropuerto[i] != NULL) {
						mapaAeropuertos[seq_id[i]] = aeropuerto[i];
					}
				}

				char resultados[10][256]; // array de hasta 10 resultados cada uno de hasta 256 caracteres
				int numResultados = 0; // contador de cuantos resultados enviamos al cloud

				// Imprimimos el histograma filtrando por umbral
				for (int i = 0; i <= maxID; i++) {
					if (h_contadores[i] >= umbral && mapaAeropuertos[i] != NULL) {
						int barras = h_contadores[i] / 1000; // escala para que no sea demasiado larga
						printf("%-6s (%7d) | %6d  | ", mapaAeropuertos[i], i, h_contadores[i]);
						for (int b = 0; b < barras; b++) {
							printf("#");
						}
						printf("\n");

						// Guardar los resultados para mandarlos al cloud, limitando a maximo 10 para no sobrecargar la base de datos
						if (numResultados < 10) {
							snprintf(resultados[numResultados], sizeof(resultados[numResultados]), "Aeropuerto %s (id = %d) - %d vuelos", mapaAeropuertos[i], i, h_contadores[i]);
							numResultados++;
						}
					}
				}

				// Calculamos el total de aeropuertos que hemos mostrado por pantalla y cumplen las condiciones dadas
				int aeropuertosMostrados = 0;
				for (int i = 0; i <= maxID; i++) {
					if (h_contadores[i] >= umbral && mapaAeropuertos[i] != NULL) {
						aeropuertosMostrados++;
					}
				}
				printf("\nAeropuertos mostrados (con mas de %d vuelos): %d (del total %d)\n", umbral, aeropuertosMostrados, aeropuertosUnicos);
				printf("\nEscala: cada # = 1000 vuelos\n");

				// ============================================================
				// CLOUD: preguntar si subir resultados
				// ============================================================
				// Solo enviamos al cloud si hay resultados que cumplen el umbral
				if (numResultados > 0) {
					char nombreFase[100]; // buffer donde guardaremos el nombre que identifica la fase
					char opcionesEntrada[256]; // buffer donde guardamos los parametros de entrada

					// Construimos el nombre y las opciones con los datos del histograma
					snprintf(nombreFase, sizeof(nombreFase), "Fase 4 - Histograma Aeropuertos %s", tipo);
					snprintf(opcionesEntrada, sizeof(opcionesEntrada), "Umbral = %d, Vuelos encontrados = %d (de %d)", umbral, aeropuertosMostrados, aeropuertosUnicos);

					// Llamamos a la funcion que pregunta al usuario si mandar los resultados al cloud
					preguntar_y_enviar(nombreFase, opcionesEntrada, resultados, numResultados);
				}
				else {
					printf("\nNo hay aeropuertos que cumplan el umbral.\n");
				}

				// Liberamos memoria
				free(mapaAeropuertos);
				free(h_contadores);
				cudaFree(dev_datos);
				cudaFree(dev_contador);

				break;
			}

			case 3: {
				int volver = 0; // para volver al menu anterior
				int opcion2 = -1;
				while (opcion2 != 0) {
					printf("\n1. Aeropuerto de Origen\n");
					printf("2. Aeropuerto de Destino\n");
					printf("3. Volver Atras\n");
					printf("Seleccione tipo de aeropuerto: ");

					// Validamos la respuesta del usuario
					if (scanf("%d", &opcion2) != 1) {
						printf("Entrada no valida.\n");
						int c;
						while ((c = getchar()) != '\n' && c != EOF);
						continue;
					}

					switch (opcion2) {
						case 1:
						case 2: {
							char codigo[10]; // buffer donde guardamos el codigo que introduce el usuario
							printf("\nIntroduce el codigo del aeropuerto (ej: ATL): ");
							if (scanf("%9s", codigo) != 1) {
								printf("Entrada no valida.\n");
								int c;
								while ((c = getchar()) != '\n' && c != EOF);
								continue;
							}

							// Buscamos el id correspondiente al codigo introducido
							int idBuscado = -1;
							for (int i = 0; i < numVuelos; i++) {
								// Con strcmp comparamos que la posicion en la que estamos (el codigo del aeropuerto) sea igual al codigo que introdujo el usuario
								if (opcion2 == 1 && origin_airport[i] != NULL && strcmp(origin_airport[i], codigo) == 0) {
									idBuscado = origin_seq_id[i];
									break;
								}
								else if (opcion2 == 2 && dest_airport[i] != NULL && strcmp(dest_airport[i], codigo) == 0) {
									idBuscado = dest_seq_id[i];
									break;
								}
							}
							if (idBuscado == -1) {
								printf("\nAeropuerto %s no encontrado\n", codigo);
								break;
							}

							// Contamos el numero de vuelos que tiene el aeropuerto del codigo introducido
							int vuelos = 0;
							for (int i = 0; i < numVuelos; i++) {
								if (opcion2 == 1 && origin_seq_id[i] == idBuscado) {
									vuelos++;
								}
								else if (opcion2 == 2 && dest_seq_id[i] == idBuscado) {
									vuelos++;
								}
							}

							// Porcentaje que representa los vuelos obtenidos 
							float porcentaje = (float)vuelos / numVuelos * 100.0f;
							printf("\nAeropuerto %s - Vuelos: %d (%.2f%% del total)\n", codigo, vuelos, porcentaje);

							// ============================================================
							// CLOUD: preguntar si subir resultados
							// ============================================================
							char nombreFase[100]; // buffer donde guardaremos el nombre que identifica la fase
							char opcionesEntrada[256]; // buffer donde guardamos los parametros de entrada
							char resultados[10][256]; // array donde guardamos hasta 10 resultados de hasta 256 caracteres
							const char* tipo = (opcion2 == 1) ? "origen" : "destino"; // identificamos el tipo de aeropuerto solicitado por el usuario

							// Construimos el nombre, opciones y unico resultado con los datos del aeropuerto encontrado
							snprintf(nombreFase, sizeof(nombreFase), "Fase 4 - Busqueda Aeropuerto %s", tipo);
							snprintf(opcionesEntrada, sizeof(opcionesEntrada), "Codigo = %s, Total vuelos = %d", codigo, numVuelos);
							snprintf(resultados[0], sizeof(resultados[0]), "Aeropuerto %s - %d vuelos (%.2f%% del total)", codigo, vuelos, porcentaje);

							// Enviamos solo 1 resultado porque es la busqueda de un unico aeropuerto
							preguntar_y_enviar(nombreFase, opcionesEntrada, resultados, 1);

							break;
						}

						// Salimos al menu anterior
						case 3: 
							volver = 1;
							opcion2 = 0;
							break;

						default:
							printf("Opcion no valida, introduzca un numero entre 1 y 3\n");
					}

					if (volver) {
						break;
					}
				}

				break;
			}

			case 4:
			case 5: {
				int volver = 0;
				int opcion2 = -1;
				while (opcion2 != 1 && opcion2 != 2 && opcion2 != 3) {
					printf("\n1. Aeropuerto de Origen\n");
					printf("2. Aeropuerto de Destino\n");
					printf("3. Volver Atras\n");
					printf("Seleccione una opcion: ");

					// Validamos la respuesta del usuario
					if (scanf("%d", &opcion2) != 1) {
						printf("Entrada no valida.\n");
						int c;
						while ((c = getchar()) != '\n' && c != EOF);
						continue;
					}

					if (opcion2 == 3) {
						volver = 1;
						break;
					}

					if (opcion2 != 1 && opcion2 != 2 && opcion2 != 3) {
						printf("Opcion no valida, introduzca un numero entre 1 y 3\n");
					}
				}

				if (volver) {
					opcion2 = 0;
					break;
				}

				// Comprobar caracteristicas hardware
				cudaDeviceProp prop;
				cudaGetDeviceProperties(&prop, 0);
				int hilosPorBloque = prop.maxThreadsPerBlock; // maximo de hilos por bloque
				int numBloques = (numVuelos + hilosPorBloque - 1) / hilosPorBloque; // calculamos el numero de bloques

				// Necesitamos saber cual es el ID maximo para reservar espacio en memoria
				int maxID = 0;
				for (int i = 0; i < numVuelos; i++) {
					if (opcion2 == 1 && origin_seq_id[i] > maxID) {
						maxID = origin_seq_id[i];
					}
					else if (opcion2 == 2 && dest_seq_id[i] > maxID) {
						maxID = dest_seq_id[i];
					}
				}

				// Definimos el contador que en este caso sera un array
				int* h_contadores = (int*)malloc((maxID + 1) * sizeof(int));
				memset(h_contadores, 0, (maxID + 1) * sizeof(int)); // inicializamos el array con todo ceros

				int* dev_datos, * dev_contador; // definimos los punteros

				// Reservamos el espacio en memoria
				cudaMalloc((void**)&dev_datos, numVuelos * sizeof(int));
				cudaMalloc((void**)&dev_contador, (maxID + 1) * sizeof(int));

				// Definiminos los punteros en funcion de la eleccion del usuario
				char** aeropuerto = (opcion2 == 1) ? origin_airport : dest_airport;
				int* seq_id = (opcion2 == 1) ? origin_seq_id : dest_seq_id;

				// Transferimos los datos de la CPU a la GPU 
				cudaMemcpy(dev_datos, seq_id, numVuelos * sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_contador, h_contadores, (maxID + 1) * sizeof(int), cudaMemcpyHostToDevice);

				// Ejecutamos el kernel adaptandonos a las caracteristicas del hardware de cada ordenador
				dim3 dimGrid(numBloques);
				dim3 dimBlock(hilosPorBloque);
				fase4_datos << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_contador);
				cudaDeviceSynchronize(); // esperamos a que todos los hilos terminen antes de continuar

				// Transferimos los datos de la GPU a la CPU 
				cudaMemcpy(h_contadores, dev_contador, (maxID + 1) * sizeof(int), cudaMemcpyDeviceToHost);

				// Relacionamos los ids de los aeropuertos con su codigo
				char** mapaAeropuertos = (char**)malloc((maxID + 1) * sizeof(char*));
				memset(mapaAeropuertos, 0, (maxID + 1) * sizeof(char*));
				for (int i = 0; i < numVuelos; i++) {
					if (seq_id[i] > 0 && aeropuerto[i] != NULL) {
						mapaAeropuertos[seq_id[i]] = aeropuerto[i];
					}
				}

				// Inicializamos los arrays donde guardaremos el top 5 de vuelos
				int topID[5] = { -1, -1, -1, -1, -1 };
				int topVuelos[5] = { 0, 0, 0, 0, 0 };
				for (int j = 0; j < 5; j++) {
					topVuelos[j] = (opcion == 4) ? 0 : INT_MAX;
				}

				// Buscamos el top 5 comparando cada aeropuerto con los que ya tenemos en el top
				for (int i = 0; i <= maxID; i++) {
					int valor = h_contadores[i];
					if (mapaAeropuertos[i] == NULL || h_contadores[i] == 0) continue;
					for (int j = 0; j < 5; j++) {
						bool condicion = (opcion == 4) ? (valor > topVuelos[j]) : (valor < topVuelos[j]);
						if (topID[j] == -1 || condicion) {
							for (int k = 4; k > j; k--) {
								topID[k] = topID[k - 1];
								topVuelos[k] = topVuelos[k - 1];
							}

							topID[j] = i;
							topVuelos[j] = valor;
							break;
						}
					}
				}

				printf("\n");
				// ============================================================
				// CLOUD: preguntar si subir resultados
				// ============================================================
				
				char resultados[10][256]; // array donde guardaremos los aeropuertos del top

				//Imprimimos el histograma filtrando por umbral
				for (int j = 0; j < 5; j++) {
					int barras = topVuelos[j] / 1000; // escala para que no sea demasiado larga
					printf("%-6s (%7d) | %6d  | ", mapaAeropuertos[topID[j]], topID[j], topVuelos[j]);
					for (int b = 0; b < barras; b++) {
						printf("#");
					}
					printf("\n");

					// Para cada aeropuerto gaurdamos un resultado con su posicion, codigo, id y numero de vuelos
					snprintf(resultados[j], sizeof(resultados[j]), "%d. Aeropuerto %s (id = %d) - %d vuelos", j + 1, mapaAeropuertos[topID[j]], topID[j], topVuelos[j]);
				}
				printf("\nEscala: cada # = 1000 vuelos\n");

				char nombreFase[100]; // buffer donde guardamos el nombre que identifica la fase
				char opcionesEntrada[256]; // buffer donde guardamos los parametros de entrada
				const char* tipoTop = (opcion == 4) ? "Mas" : "Menos"; // identificamos si el top de mas o menos concurridos
				const char* tipoAero = (opcion2 == 1) ? "origen" : "destino"; // identificamos si es aeropuerto de origen o de destino

				// Construimos el nombre y las opciones con los datos del top
				snprintf(nombreFase, sizeof(nombreFase), "Fase 4 - Top 5 %s Concurridos (%s)", tipoTop, tipoAero);
				snprintf(opcionesEntrada, sizeof(opcionesEntrada), "Tipo = %s, Criterio = %s, Total vuelos = %d", tipoAero, tipoTop, numVuelos);

				// Enviamos los 5 resultados del top
				preguntar_y_enviar(nombreFase, opcionesEntrada, resultados, 5);

				// Liberamos memoria
				free(mapaAeropuertos);
				free(h_contadores);
				cudaFree(dev_datos);
				cudaFree(dev_contador);

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
#include <cuda_runtime.h> // para funciones Cuda
#include <stdio.h> // para printf y scanf
#include <math.h> // para que NAN funcione
#include <string.h> // para que strdup funcione
#include <stdlib.h>  // para malloc, atoi, atof
#include <windows.h> // para Sleep

// Tenemos que incluir las cabeceras de los archivos donde imlementaremos las fases
#include "fase1.cuh"
#include "fase2.cuh"
#include "fase3.cuh"
#include "fase4.cuh"
#include "cloud.cuh"

#define MAX_VUELOS 1200000
#define RUTA "C:\\Airline_dataset.csv"

// Creamos una estructura donde agrupemos todos los arrays que contienen los datos del CSV en una misma estructura
struct Dataset {
	int numVuelos; // numero total de vuelos
	char** tail_num; // columna 3
	int* origin_seq_id; // columna 5
	char** origin_airport; // columna 6
	int* dest_seq_id; // columna 7
	char** dest_airport; // columna 8
	float* dep_time; // columna 9
	float* dep_delay; // columna 10
	float* arr_time; // columna 11
	float* arr_delay; // columna 12
	float* weather_delay; // columna 13
};

// Definimos la funcion donde vamos a hacer la lectura del csv
Dataset* leerCSV(const char* ruta) {
	// Abrir el fichero CSV
	FILE* fichero = fopen(ruta, "r");

	// Compruebo que el fichero se ha abierto correctamente
	if (fichero == NULL) {
		return NULL;
	}

	char linea[512]; // buffer donde guardamos cada linea del CSV al leerla

	fgets(linea, 512, fichero); // Saltar la cabecera

	// Vamos a reservar acceso en memoria
	Dataset* ds = (Dataset*)malloc(sizeof(Dataset));

	// Reservamos memoria para cada array del dataset
	ds->tail_num = (char**)malloc(MAX_VUELOS * sizeof(char*));
	ds->origin_seq_id = (int*)malloc(MAX_VUELOS * sizeof(int));
	ds->origin_airport = (char**)malloc(MAX_VUELOS * sizeof(char*));
	ds->dest_seq_id = (int*)malloc(MAX_VUELOS * sizeof(int));
	ds->dest_airport = (char**)malloc(MAX_VUELOS * sizeof(char*)); // char* porque son array de punteros
	ds->dep_time = (float*)malloc(MAX_VUELOS * sizeof(float));
	ds -> dep_delay = (float*)malloc(MAX_VUELOS * sizeof(float));
	ds->arr_time = (float*)malloc(MAX_VUELOS * sizeof(float));
	ds->arr_delay = (float*)malloc(MAX_VUELOS * sizeof(float));
	ds->weather_delay = (float*)malloc(MAX_VUELOS * sizeof(float));

	// Leer el fichero
	int fila = 0;
	while (fgets(linea, 512, fichero) && fila < MAX_VUELOS) {
		int i = 0; // contador de campos que se reinicia en cada linea
		char* pos = linea; // puntero con el que apuntamos al inicio de ela linea y avanzamos campo por campo
		while (pos != NULL && *pos != '\0' && *pos != '\n') { // el bucle avanza desde donde el puntero esta en la posicion cero hasta donde el puntero esta al final de linea
			char* coma = strchr(pos, ','); // buscamos la siguiente coma desde la posicion actual
			char campo[128] = ""; // buffer donde guardamos el contenido del campo actual, inicializado vacio para evitar datos basura
			if (coma != NULL) {
				int len = (int)(coma - pos); // calculamos la longitud del campo, es decir, cuantos caracteres hay entre la posicion actual y la coma
				if (len > 0 && len < 128) {
					strncpy(campo, pos, len); // copiamos los caracteres seleccionados en campo
					campo[len] = '\0';
				}
				pos = coma + 1; // avanzamos al siguiente campo
			}
			else {
				// Ultimo campo de la linea (no tiene coma despues)
				strncpy(campo, pos, 127);
				campo[127] = '\0';  // terminamos el string
				campo[strcspn(campo, "\r\n")] = '\0'; // Eliminar salto de linea del ultimo campo
				pos = NULL; // terminamos el bucle poniendo el puntero a NULL
			}

			int vacio = (strlen(campo) == 0); // comprobamos si el campo esta vacio

			// En funcion del indice del campo guardamos el valor en el array correspondiente del dataset.Si el campo esta vacio guardamos 0 en int, NAN en float y NULL en string
			if (i == 3) {
				ds->tail_num[fila] = vacio ? NULL : strdup(campo);
			}
			else if (i == 5) {
				ds->origin_seq_id[fila] = vacio ? 0 : (int)atof(campo);
			}
			else if (i == 6) {
				ds->origin_airport[fila] = vacio ? NULL : strdup(campo);
			}
			else if (i == 7) {
				ds->dest_seq_id[fila] = vacio ? 0 : (int)atof(campo);
			}
			else if (i == 8) {
				ds->dest_airport[fila] = vacio ? NULL : strdup(campo);
			}
			else if (i == 9) {
				ds->dep_time[fila] = vacio ? NAN : (float)atof(campo);
			}
			else if (i == 10) {
				ds->dep_delay[fila] = vacio ? NAN : (float)atof(campo);
			}
			else if (i == 11) {
				ds->arr_time[fila] = vacio ? NAN : (float)atof(campo);
			}
			else if (i == 12) {
				ds->arr_delay[fila] = vacio ? NAN : (float)atof(campo);
			}
			else if (i == 13) {
				ds->weather_delay[fila] = vacio ? NAN : (float)atof(campo);
			}
			i++; // incrementamos el contador
		}
		fila++; // incrementamos el numero de fila
	}

	// Guardamos el numero de vuelos leidos
	ds->numVuelos = fila;

	// Cerrar el fichero
	fclose(fichero);

	return ds; // porque la funcion devuelve Dataset*
}

// Funcion que libera toda la memoria reservada por leerCSV
void liberarDataset(Dataset* ds) {
	if (ds == NULL) return; // si el dataset esta vacio pues ya esta

	// Liberamos cada string que reservamos con strdup
	for (int i = 0; i < ds->numVuelos; i++) {
		if (ds->tail_num[i]) {
			free(ds->tail_num[i]);
		}
		if (ds->origin_airport[i]) {
			free(ds->origin_airport[i]);
		}
		if (ds->dest_airport[i]) {
			free(ds->dest_airport[i]);
		}
	}

	// Liberamos los arrays
	free(ds->tail_num);
	free(ds->origin_seq_id);
	free(ds->origin_airport);
	free(ds->dest_seq_id);
	free(ds->dest_airport);
	free(ds->dep_time);
	free(ds->dep_delay);
	free(ds->arr_time);
	free(ds->arr_delay);
	free(ds->weather_delay);

	// Liberamos la estructura
	free(ds);
}

int main() {
	// Mostramos el titulo de la practica durante 1.5 segundos antes de continuar
	printf("======================================================================================================================\n");
	printf("						EL2 PAP 2026						  \n");
	printf("				Candela Sanz Martin y Maria de la Orden Montes					\n");
	printf("======================================================================================================================\n"); 
	printf("\nBienvenido!\n");
	Sleep(1500);

	// Pedimos la ruta al usuario
	Dataset* ds = NULL;
	while (ds == NULL) {
		printf("\nIntroduzca la ruta base del dataset: ");
		printf("\n(pulse Intro para usar por defecto: C:\\Airline_dataset.csv)\n");
		char ruta[256];
		fgets(ruta, 256, stdin);
		ruta[strcspn(ruta, "\n")] = '\0'; // Eliminar el salto de linea que mete fgets
		// Si pulsa Intro sin escribir nada, usar ruta por defecto
		if (strlen(ruta) == 0) {
			strcpy(ruta, RUTA);
		}

		ds = leerCSV(ruta);
		// Contemplamos la opcion de que el dataset no se cargue bien
		if (ds == NULL) {
			printf("Error al cargar el dataset, intentarlo de nuevo\n");
		}
	}

	printf("Dataset cargado: %d vuelos\n", ds->numVuelos);

	// Menu
	int opcion = -1;
	while (opcion != 0) {
		printf("\nMenu:");
		printf("\n1. Despegues");
		printf("\n2. Aterrizajes");
		printf("\n3. Retrasos");
		printf("\n4. Histograma de Aeropuertos");
		printf("\n5. Modificar Ruta Base");
		printf("\n6. Salir");
		printf("\nElija una opcion: ");

		// Validamos la respuesta del usuario
		if (scanf("%d", &opcion) != 1) {
			printf("Entrada no valida.\n");
			int c;
			while ((c = getchar()) != '\n' && c != EOF);
			continue;
		}

		switch (opcion) {
			case 1: 
				ejecutarFase1(ds -> dep_delay, ds -> numVuelos);
				break;
			case 2: 
				ejecutarFase2(ds->arr_delay, ds->numVuelos, ds->tail_num);
				break;
			case 3: 
				ejecutarFase3(ds->dep_delay, ds->arr_delay, ds->weather_delay, ds->dep_time, ds->arr_time, ds->numVuelos);
				break;
			case 4: 
				ejecutarFase4(ds->origin_seq_id, ds->dest_seq_id, ds->origin_airport, ds->dest_airport, ds->numVuelos);
				break;
			case 5: {
				// Pedimos la nueva ruta al usuario y cargamos el dataset
				Dataset* ds_nuevo = NULL;
				while (ds_nuevo == NULL) {
					printf("\nIntroduzca la ruta base del dataset: ");
					printf("\n(pulse Intro para usar por defecto: C:\\Airline_dataset.csv)\n");
					char ruta[256];

					// Limpiamos el buffer antes de leer la nueva ruta
					int c;
					while ((c = getchar()) != '\n' && c != EOF);

					fgets(ruta, 256, stdin);
					ruta[strcspn(ruta, "\n")] = '\0'; // Eliminar el salto de linea que mete fgets
					// Si pulsa Intro sin escribir nada, usar ruta por defecto
					if (strlen(ruta) == 0) {
						strcpy(ruta, RUTA);
					}

					ds_nuevo = leerCSV(ruta);
					// Contemplamos la opcion de que el dataset no se cargue bien
					if (ds_nuevo == NULL) {
						printf("Error al cargar el dataset, intentarlo de nuevo\n");
					}
				}

				liberarDataset(ds); // liberamos el dataset anterior
				ds = ds_nuevo; // asignamos el nuevo dataset

				printf("Dataset cargado: %d vuelos\n", ds->numVuelos);

				break;
			}

			case 6: 
				printf("Hasta luego!\n");
				opcion = 0;
				break;

			default: 
				printf("Opcion no valida, introduzca un numero entre 1 y 6\n"); 
				break;
		}
	}
	
	liberarDataset(ds);
	return 0;
}
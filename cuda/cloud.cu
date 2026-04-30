#include "cloud.cuh"
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

#define MAX_RESULTADOS 10

void enviar_al_cloud(const char* fase, const char* opciones, const char* resultadosArray[], int numResultados) {
	if (fase == NULL || opciones == NULL) {
		printf("ERROR: faltan parametros obligatorios para el envio al cloud\n");
		return;
	}

	if (numResultados > MAX_RESULTADOS) {
		printf("AVISO: se recortan los resultados a %d (limite del sistema).\n", MAX_RESULTADOS);
		numResultados = MAX_RESULTADOS;
	}

	if (numResultados < 0) {
		numResultados = 0;
	}

	char usuario[100];
	char json[8192];

	int valido = 0;
	while (!valido) {
		printf("Introduce tu nombre de usuario: ");
		if (scanf("%99s", usuario) != 1) {
			printf("ERROR: entrada no valida. Intenta de nuevo.\n");
			continue;
		}
		if (strlen(usuario) == 0) {
			printf("ERROR: el nombre no puede estar vacío.\n");
			continue;
		}
		valido = 1;
	}

	// Construir el JSON
	char arrayResultados[4096] = "[";
	for (int i = 0; i < numResultados; i++) {
		char temp[300];
		if (i > 0) strcat(arrayResultados, ",");
		snprintf(temp, sizeof(temp), "\"%s\"", resultadosArray[i]);
		strcat(arrayResultados, temp);
	}

	strcat(arrayResultados, "]");

	snprintf(json, sizeof(json),
		"{\"usuario\":\"%s\",\"fase\":\"%s\",\"opciones_entrada\":\"%s\",\"resultados\":%s}",
		usuario, fase, opciones, arrayResultados);

	// Inicializar el curl
	CURL* curl = curl_easy_init();
	if (!curl) {
		printf("ERROR: No se pudo inicializar curl.\n");
		return;
	}

	// Configurar headers
	struct curl_slist* headers = NULL;
	headers = curl_slist_append(headers, "Content-Type: application/json");
	if (!headers) {
		printf("ERROR: no se pudo crear la cabecera HTTP.\n");
		curl_easy_cleanup(curl);
		return;
	}

	// Configurar la petición (COMENTAR BIEN BIEN)
	curl_easy_setopt(curl, CURLOPT_URL, "https://pl2-pap.azurewebsites.net/api/partidas");
	curl_easy_setopt(curl, CURLOPT_POST, 1L);
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json);
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

	// Ejecutar
	printf("\nEnviando datos al cloud... \n");
	CURLcode res = curl_easy_perform(curl);

	if (res != CURLE_OK) {
		fprintf(stderr, "ERROR %s\n", curl_easy_strerror(res));
	}
	else {
		long codigoHTTP = 0;
		curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &codigoHTTP);

		if (codigoHTTP == 201) {
			printf("\nPartida registrada en el cloud (HTTP 201).\n");
		}
		else if (codigoHTTP == 400) {
			printf("\nERROR: datos invalidos enviados al servidor (HTTP 400).\n");
		}
		else if (codigoHTTP == 500) {
			printf("\nERROR: error en el servidor (HTTP 500). Intentalo mas tarde.\n");
		}
		else {
			printf("\nAVISO: Servidor no respondio con codigo HTTP inesperado: %ld\n", codigoHTTP);
		}
	}

	// Liberar recursos
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);
}


void preguntar_y_enviar(const char* fase, const char* opciones, char resultados[][256], int numResultados) {
	if (fase == NULL || opciones == NULL) {
		printf("ERROR: datos invalidos al intentar subir al cloud\n");
		return;
	}

	char respuesta;
	int valida = 0;

	// Limpiar buffer
	int c;
	while ((c = getchar()) != '\n' && c != EOF);

	while (!valida) {
		printf("\nQuieres enviar los resultados al cloud (s/n): ");

		if (scanf(" %c", &respuesta) != 1) {
			printf("Entrada no valida. Intentalo de nuevo.\n");
			while ((c = getchar()) != '\n' && c != EOF);
			continue;
		}

		if (respuesta == 's' || respuesta == 'S' || respuesta == 'n' || respuesta == 'N') {
			valida = 1;
		}
		else {
			printf("Opcion no valida. Introduce 's' o 'n'.\n");
			while ((c = getchar()) != '\n' && c != EOF);
		}
	}

	if (respuesta == 's' || respuesta == 'S') {
		const char* arrayPtr[10];
		int n = (numResultados > MAX_RESULTADOS) ? MAX_RESULTADOS : numResultados;
		for (int i = 0; i < n; i++) {
			arrayPtr[i] = resultados[i];
		}
		enviar_al_cloud(fase, opciones, arrayPtr, n);
	}
	else {
		printf("Datos no enviados al cloud\n");
	}
}
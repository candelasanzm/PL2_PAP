#include "cloud.cuh"
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

void enviar_al_cloud(const char* fase, const char* opciones, const char* resultado) {
	char usuario[100];
	char json[2048];

	// Pedir nombre al usuario
	printf("Introduce tu nombre de usuario: ");
	scanf("%99s", usuario);

	// Construir el JSON
	snprintf(json, sizeof(json),
		"{\"usuario\":\"%s\",\"fase\":\"%s\",\"opciones_entrada\":\"%s\",\"resultado\":\"%s\"}",
		usuario, fase, opciones, resultado);

	// Inicializar el curl
	CURL* curl = curl_easy_init();
	if (!curl) {
		printf("ERROR: No se pudo inicializar curl.\n");
		return;
	}

	// Configurar headers
	struct curl_slist* headers = NULL;
	headers = curl_slist_append(headers, "Content-Type: application/json");

	// Configurar la petición (COMENTAR BIEN BIEN)
	curl_easy_setopt(curl, CURLOPT_URL, "https://pl2-pap.azurewebsites.net/api/partidas");
	curl_easy_setopt(curl, CURLOPT_POST, 1L);
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json);
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

	// Ejecutar
	printf("\nEnviando datos al cloud... \n");
	CURLcode res = curl_easy_perform(curl);

	if (res != CURLE_OK) {
		fprintf(stderr, "ERROR %s\n", curl_easy_strerror(res));
	}
	else {
		printf("\nDatos enviados correctamente.\n");
	}

	// Liberar recursos
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);
}
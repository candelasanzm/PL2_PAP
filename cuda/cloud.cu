#include "cloud.cuh" // declaraciones de las funciones del cloud
#include <stdio.h> // para printf y scanf
#include <string.h> // para strcat, strlen y snprintf
#include <curl/curl.h> // libreria libcurl que usamos para hacer peticiones HTTP

#define MAX_RESULTADOS 10 // limite maximo de resultados que enviar al cloud para no sobrecargar la base de datos 

// Funcion que envia los datos al cloud
void enviar_al_cloud(const char* fase, const char* opciones, const char* resultadosArray[], int numResultados) {
	// Comprobamos que los parametros obligatorios no sean nulos
	if (fase == NULL || opciones == NULL) {
		printf("ERROR: faltan parametros obligatorios para el envio al cloud\n");
		return;
	}

	// Validacion: limitamos el numero de resultados al maximo permitido
	if (numResultados > MAX_RESULTADOS) {
		printf("AVISO: se recortan los resultados a %d (limite del sistema).\n", MAX_RESULTADOS);
		numResultados = MAX_RESULTADOS;
	}

	// Validacion: si el numero es negativo lo forzamos a cero
	if (numResultados < 0) {
		numResultados = 0;
	}

	char usuario[100]; // buffer donde guardamos el numero del usuario
	char json[8192]; // buffer donde construimos el JSON completo que enviaremos al cloud


	// Pedimos el nombre de usuario validando que la entrada no este vacia y bien formada
	int valido = 0;
	while (!valido) {
		printf("Introduce tu nombre de usuario: ");
		// Validacion el scanf debe leer 1
		if (scanf("%99s", usuario) != 1) {
			printf("ERROR: entrada no valida. Intenta de nuevo.\n");
			continue;
		}
		if (strlen(usuario) == 0) { // validacion que no este vacio
			printf("ERROR: el nombre no puede estar vacio.\n");
			continue;
		}
		valido = 1;
	}

	// Construir el JSON
	char arrayResultados[4096] = "["; // abrimos el json
	for (int i = 0; i < numResultados; i++) {
		char temp[300]; // buffer temporal para escribir cada elemento del array
		// Entre elementos ponemos una coma 
		if (i > 0) {
			strcat(arrayResultados, ",");
		}
		snprintf(temp, sizeof(temp), "\"%s\"", resultadosArray[i]); // envolvemos cada resultado entre comillas
		strcat(arrayResultados, temp); // lo añadimos al array completo
	}

	strcat(arrayResultados, "]"); // cerramos el json

	// Construimos el json completo con los datos para enviar
	snprintf(json, sizeof(json),
		"{\"usuario\":\"%s\",\"fase\":\"%s\",\"opciones_entrada\":\"%s\",\"resultados\":%s}",
		usuario, fase, opciones, arrayResultados);

	// Inicializar el curl que es la libreria que hace la peticion HTTP
	CURL* curl = curl_easy_init();
	if (!curl) {
		printf("ERROR: No se pudo inicializar curl.\n");
		return;
	}

	// Configuramos las cabeceras para indicar que el contenido es json
	struct curl_slist* headers = NULL;
	headers = curl_slist_append(headers, "Content-Type: application/json");
	if (!headers) {
		printf("ERROR: no se pudo crear la cabecera HTTP.\n"); // si no se pudo crear la cabecera limpiamos curl y salimos
		curl_easy_cleanup(curl);
		return;
	}

	// Configuramos las opciones de la peticion HTTP 
	curl_easy_setopt(curl, CURLOPT_URL, "https://pl2-pap.azurewebsites.net/api/partidas"); // URL del endpoint en Azure
	curl_easy_setopt(curl, CURLOPT_POST, 1L); // indicamos que es una peticion POST
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers); // ponemos las cabeceras
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json); // adjuntamos el cuerpo (el json construido antes)
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L); // desactivamos la verificacion ssl del peer
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L); // desactivamos verificacion ssl del host
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L); // timeout de 60 segundos para evitar bloqueos largos en cold starts de Azure

	// Ejecutamos la peticion
	printf("\nEnviando datos al cloud... \n");
	CURLcode res = curl_easy_perform(curl);

	// Comprobamos si hubo error del curl al ejecutar la peticion
	if (res != CURLE_OK) {
		fprintf(stderr, "ERROR %s\n", curl_easy_strerror(res));
	}
	else {
		// Si curl ejecuto bien, comprobamos el codigo HTTP que devolvio el servidor
		long codigoHTTP = 0;
		curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &codigoHTTP);

		if (codigoHTTP == 201) { // creado correctamente
			printf("\nPartida registrada en el cloud (HTTP 201).\n");
		}
		else if (codigoHTTP == 400) { // el cliente mando datos mal formados
			printf("\nERROR: datos invalidos enviados al servidor (HTTP 400).\n");
		}
		else if (codigoHTTP == 500) { // error en el servidor
			printf("\nERROR: error en el servidor (HTTP 500). Intentalo mas tarde.\n");
		}
		else { // cualquier otro codigo no esperado
			printf("\nAVISO: Servidor no respondio con codigo HTTP inesperado: %ld\n", codigoHTTP);
		}
	}

	// Liberamos los recursos asignados por curl
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);
}

// Funcion que pregunta al usuario si quiere subir al cloud y si la respuesta es si envia los datos
void preguntar_y_enviar(const char* fase, const char* opciones, char resultados[][256], int numResultados) {
	// Validacion de parametros no nulos
	if (fase == NULL || opciones == NULL) {
		printf("ERROR: datos invalidos al intentar subir al cloud\n");
		return;
	}

	char respuesta;
	int valida = 0;

	// Limpiar buffer del scan anterior 
	int c;
	while ((c = getchar()) != '\n' && c != EOF);

	// Bucle que pide la respuesta hasta que el usuario introduzca una respuesta valida
	while (!valida) {
		printf("\nQuieres enviar los resultados al cloud (s/n): ");

		// Validamos la respuesta del usuario
		if (scanf(" %c", &respuesta) != 1) {
			printf("Entrada no valida. Intentalo de nuevo.\n");
			while ((c = getchar()) != '\n' && c != EOF);
			continue;
		}

		// Si la respuesta es valida salimos del bucle
		if (respuesta == 's' || respuesta == 'S' || respuesta == 'n' || respuesta == 'N') {
			valida = 1;
		}
		else { // si no es valida mostramos un error y volvemos a pedir una respuesta
			printf("Opcion no valida. Introduce 's' o 'n'.\n");
			while ((c = getchar()) != '\n' && c != EOF);
		}
	}

	// Si el usuario dice que no salimos sin enviar datos al cloud
	if (respuesta == 'n' || respuesta == 'N') {
		printf("Datos no enviados al cloud\n");
		return;
	}

	// Si el usuario dice que si, gestionamos cuantos resultados enviamos 
	int limite = numResultados;

	// Preguntamos al usuario cuantos resultados enviar en caso de que el numero de resultados obtenidos haya sido mayor que 5
	if (numResultados > 5) {
		int maxLimite = (numResultados > MAX_RESULTADOS) ? MAX_RESULTADOS : numResultados; // como mucho mandamos 10 resultados
		limite = -1; // forzamos a entrar en el bucle de validacion

		// Bucle hasta que el usuario meta un limite valido
		while (limite < 1 || limite > maxLimite) {
			printf("Cuantos resultados quieres guardar (1 a %d): ", maxLimite);
			if (scanf("%d", &limite) != 1) {
				printf("Entrada no valida.\n");
				while ((c = getchar()) != '\n' && c != EOF);
				limite = -1;
				continue;
			}
			if (limite < 1 || limite > maxLimite) {
				printf("ERROR: el valor debe estar entre 1 y %d.\n", maxLimite);
			}
		}
	}

	// Convetimos el array bidimensional fijo a un array de punteros para pasarlo a la funcion que envia los datos al cloud
	const char* arrayPtr[MAX_RESULTADOS];
	for (int i = 0; i < limite; i++) { 
		arrayPtr[i] = resultados[i];
	}

	// Llamamos a la funcion que hace el envio HTTP real
	enviar_al_cloud(fase, opciones, arrayPtr, limite);
}
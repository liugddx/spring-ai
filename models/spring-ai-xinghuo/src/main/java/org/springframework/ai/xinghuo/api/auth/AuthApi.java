/*
 * Copyright 2023-2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.xinghuo.api.auth;


import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Base64;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/**
 * XingHuo abstract authentication API.
 *
 * @author Guangdong Liu
 * @since 1.0
 */
public abstract class AuthApi {

	/**
	 * Generates a URL with authentication parameters.
	 *
	 * @param host       The request host, e.g., "spark-api.xf-yun.com"
	 * @param path       The request path, e.g., "/v1.1/chat"
	 * @param httpMethod The HTTP method, e.g., "POST"
	 * @param apiKey     The API Key obtained from the console
	 * @param apiSecret  The API Secret obtained from the console
	 * @return The complete URL with authentication parameters
	 * @throws Exception If an error occurs during URL generation
	 */
	public static String generateAuthUrl(String host, String path, String httpMethod, String apiKey, String apiSecret) throws Exception {
		// 1. Generate the 'date' parameter
		String date = generateDate();

		// 2. Generate the 'authorization' parameter
		String authorization = generateAuthorization(host, date, httpMethod, path, apiKey, apiSecret);

		// 3. Build the final URL with authentication parameters

        return buildUrl(host, path, authorization, date);
	}

	/**
	 * Generates the current UTC time in RFC1123 format.
	 *
	 * @return A formatted date string, e.g., "Fri, 05 May 2023 10:43:39 GMT"
	 */
	private static String generateDate() {
		SimpleDateFormat dateFormat = new SimpleDateFormat("EEE, dd MMM yyyy HH:mm:ss 'GMT'", Locale.getDefault());
		dateFormat.setTimeZone(TimeZone.getTimeZone("GMT"));
		return dateFormat.format(new Date());
	}

	/**
	 * Generates the 'authorization' parameter.
	 *
	 * @param host        The request host
	 * @param date        The 'date' parameter
	 * @param httpMethod  The HTTP method
	 * @param path        The request path
	 * @param apiKey      The API Key
	 * @param apiSecret   The API Secret
	 * @return The generated 'authorization' string
	 * @throws Exception If an error occurs during authorization generation
	 */
	private static String generateAuthorization(String host, String date, String httpMethod, String path, String apiKey, String apiSecret) throws Exception {
		// 2.1 Concatenate the 'tmp' string
		String tmp = "host: " + host + "\n"
				+ "date: " + date + "\n"
				+ httpMethod + " " + path + " HTTP/1.1";

		// 2.2 Sign the 'tmp' string using HMAC-SHA256 algorithm
		Mac mac = Mac.getInstance("HmacSHA256");
		SecretKeySpec secretKeySpec = new SecretKeySpec(apiSecret.getBytes(StandardCharsets.UTF_8), "HmacSHA256");
		mac.init(secretKeySpec);
		byte[] tmpSha = mac.doFinal(tmp.getBytes(StandardCharsets.UTF_8));

		// 2.3 Encode the signature using Base64
		String signature = Base64.getEncoder().encodeToString(tmpSha);

		// 2.4 Construct the 'authorization_origin' string
		String authorizationOrigin = String.format(
				"api_key=\"%s\", algorithm=\"hmac-sha256\", headers=\"host date request-line\", signature=\"%s\"",
				apiKey, signature
		);

		// 2.5 Encode the 'authorization_origin' string using Base64 to get the final 'authorization'

        return Base64.getEncoder().encodeToString(authorizationOrigin.getBytes(StandardCharsets.UTF_8));
	}

	/**
	 * Builds the final URL with authentication parameters.
	 *
	 * @param host          The request host
	 * @param path          The request path
	 * @param authorization The 'authorization' parameter
	 * @param date          The 'date' parameter
	 * @return The complete URL with authentication parameters
     */
	private static String buildUrl(String host, String path, String authorization, String date) {
		String baseUrl = "https://" + host + path;

        return baseUrl + "?"
                + "authorization=" + URLEncoder.encode(authorization, StandardCharsets.UTF_8)
                + "&date=" + URLEncoder.encode(date, StandardCharsets.UTF_8)
                + "&host=" + URLEncoder.encode(host, StandardCharsets.UTF_8);
	}

}

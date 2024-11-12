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

package org.springframework.ai.xinghuo;

import org.springframework.ai.xinghuo.api.XingHuoApi;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.util.StringUtils;

/**
 * @author Guangdong Liu
 */
@SpringBootConfiguration
public class XingHuoTestConfiguration {

	@Bean
	public XingHuoApi XingHuoApi() {
		return new XingHuoApi(getApiKey(), getSecretKey());
	}

	private String getApiKey() {
		String apiKey = System.getenv("XingHuo_API_KEY");
		if (!StringUtils.hasText(apiKey)) {
			throw new IllegalArgumentException(
					"You must provide an API key. Put it in an environment variable under the name XingHuo_API_KEY");
		}
		return apiKey;
	}

	private String getSecretKey() {
		String apiKey = System.getenv("XingHuo_SECRET_KEY");
		if (!StringUtils.hasText(apiKey)) {
			throw new IllegalArgumentException(
					"You must provide a secret key. Put it in an environment variable under the name XingHuo_SECRET_KEY");
		}
		return apiKey;
	}

	@Bean
	public XingHuoChatModel XingHuoChatModel(XingHuoApi api) {
		return new XingHuoChatModel(api);
	}

}

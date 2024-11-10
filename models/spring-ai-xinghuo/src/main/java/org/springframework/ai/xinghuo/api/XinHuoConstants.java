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

package org.springframework.ai.xinghuo.api;

import org.springframework.ai.observation.conventions.AiProvider;

/**
 * The ApiUtils class provides utility methods for working with API requests and
 * responses.
 *
 * @author Geng Rong
 * @since 1.0
 */
public final class XinHuoConstants {

	public static final String DEFAULT_BASE_URL = "https://aip.baidubce.com/rpc/2.0/ai_custom";

	public static final String PROVIDER_NAME = AiProvider.QIANFAN.value();

	private XinHuoConstants() {

	}

}

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

package org.springframework.ai.xinghuo.metadata;

import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.xinghuo.api.XingHuoApi;
import org.springframework.util.Assert;

/**
 * {@link Usage} implementation for {@literal XingHuo}.
 *
 * @author Thomas Vitale
 */
public class XingHuoUsage implements Usage {

	private final XingHuoApi.Usage usage;

	protected XingHuoUsage(XingHuoApi.Usage usage) {
		Assert.notNull(usage, "XingHuo Usage must not be null");
		this.usage = usage;
	}

	public static XingHuoUsage from(XingHuoApi.Usage usage) {
		return new XingHuoUsage(usage);
	}

	protected XingHuoApi.Usage getUsage() {
		return this.usage;
	}

	@Override
	public Long getPromptTokens() {
		Integer promptTokens = getUsage().promptTokens();
		return promptTokens != null ? promptTokens.longValue() : 0;
	}

	@Override
	public Long getGenerationTokens() {
		Integer generationTokens = getUsage().completionTokens();
		return generationTokens != null ? generationTokens.longValue() : 0;
	}

	@Override
	public Long getTotalTokens() {
		Integer totalTokens = getUsage().totalTokens();
		if (totalTokens != null) {
			return totalTokens.longValue();
		}
		return getPromptTokens() + getGenerationTokens();
	}

	@Override
	public String toString() {
		return getUsage().toString();
	}

}

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

package org.springframework.ai.xinghuo.image;

import io.micrometer.observation.tck.TestObservationRegistry;
import io.micrometer.observation.tck.TestObservationRegistryAssert;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariables;

import org.springframework.ai.image.ImagePrompt;
import org.springframework.ai.image.ImageResponse;
import org.springframework.ai.image.observation.DefaultImageModelObservationConvention;
import org.springframework.ai.observation.conventions.AiOperationType;
import org.springframework.ai.observation.conventions.AiProvider;
import org.springframework.ai.qianfan.XingHuoImageModel;
import org.springframework.ai.qianfan.XingHuoImageOptions;
import org.springframework.ai.qianfan.api.XinHuoImageApi;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Bean;
import org.springframework.retry.support.RetryTemplate;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.ai.image.observation.ImageModelObservationDocumentation.HighCardinalityKeyNames;
import static org.springframework.ai.image.observation.ImageModelObservationDocumentation.LowCardinalityKeyNames;

/**
 * Integration tests for observation instrumentation in {@link XingHuoImageModel}.
 *
 * @author Geng Rong
 */
@SpringBootTest(classes = XingHuoImageModelObservationIT.Config.class)
@EnabledIfEnvironmentVariables({ @EnabledIfEnvironmentVariable(named = "QIANFAN_API_KEY", matches = ".+"),
		@EnabledIfEnvironmentVariable(named = "QIANFAN_SECRET_KEY", matches = ".+") })
public class XingHuoImageModelObservationIT {

	@Autowired
	TestObservationRegistry observationRegistry;

	@Autowired
    XingHuoImageModel imageModel;

	@Test
	void observationForImageOperation() {
		var options = XingHuoImageOptions.builder()
			.withModel(XinHuoImageApi.ImageModel.Stable_Diffusion_XL.getValue())
			.withHeight(1024)
			.withWidth(1024)
			.withStyle("Base")
			.build();

		var instructions = "Here comes the sun";

		ImagePrompt imagePrompt = new ImagePrompt(instructions, options);

		ImageResponse imageResponse = this.imageModel.call(imagePrompt);
		assertThat(imageResponse.getResults()).hasSize(1);

		TestObservationRegistryAssert.assertThat(this.observationRegistry)
			.doesNotHaveAnyRemainingCurrentObservation()
			.hasObservationWithNameEqualTo(DefaultImageModelObservationConvention.DEFAULT_NAME)
			.that()
			.hasContextualNameEqualTo("image " + XinHuoImageApi.ImageModel.Stable_Diffusion_XL.getValue())
			.hasLowCardinalityKeyValue(LowCardinalityKeyNames.AI_OPERATION_TYPE.asString(),
					AiOperationType.IMAGE.value())
			.hasLowCardinalityKeyValue(LowCardinalityKeyNames.AI_PROVIDER.asString(), AiProvider.QIANFAN.value())
			.hasLowCardinalityKeyValue(LowCardinalityKeyNames.REQUEST_MODEL.asString(),
					XinHuoImageApi.ImageModel.Stable_Diffusion_XL.getValue())
			.hasHighCardinalityKeyValue(HighCardinalityKeyNames.REQUEST_IMAGE_SIZE.asString(), "1024x1024")
			.hasHighCardinalityKeyValue(HighCardinalityKeyNames.REQUEST_IMAGE_STYLE.asString(), "Base")
			.hasBeenStarted()
			.hasBeenStopped();
	}

	@SpringBootConfiguration
	static class Config {

		@Bean
		public TestObservationRegistry observationRegistry() {
			return TestObservationRegistry.create();
		}

		@Bean
		public XinHuoImageApi qianFanImageApi() {
			return new XinHuoImageApi(System.getenv("QIANFAN_API_KEY"), System.getenv("QIANFAN_SECRET_KEY"));
		}

		@Bean
		public XingHuoImageModel qianFanImageModel(XinHuoImageApi xinHuoImageApi,
                                                   TestObservationRegistry observationRegistry) {
			return new XingHuoImageModel(xinHuoImageApi, XingHuoImageOptions.builder().build(),
					RetryTemplate.defaultInstance(), observationRegistry);
		}

	}

}

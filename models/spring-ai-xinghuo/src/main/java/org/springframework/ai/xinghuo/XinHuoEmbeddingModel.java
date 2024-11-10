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

package org.springframework.ai.qianfan;

import java.util.List;

import io.micrometer.observation.ObservationRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.ai.document.Document;
import org.springframework.ai.document.MetadataMode;
import org.springframework.ai.embedding.AbstractEmbeddingModel;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.embedding.EmbeddingResponseMetadata;
import org.springframework.ai.embedding.observation.DefaultEmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationContext;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationDocumentation;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.qianfan.api.XinHuoApi;
import org.springframework.ai.qianfan.api.XinHuoApi.EmbeddingList;
import org.springframework.ai.qianfan.api.XinHuoConstants;
import org.springframework.ai.qianfan.metadata.XinHuoUsage;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.lang.Nullable;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;

/**
 * QianFan Embedding Client implementation.
 *
 * @author Geng Rong
 * @author Thomas Vitale
 * @since 1.0
 */
public class XinHuoEmbeddingModel extends AbstractEmbeddingModel {

	private static final Logger logger = LoggerFactory.getLogger(XinHuoEmbeddingModel.class);

	private static final EmbeddingModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultEmbeddingModelObservationConvention();

	private final org.springframework.ai.qianfan.XinHuoEmbeddingOptions defaultOptions;

	private final RetryTemplate retryTemplate;

	private final XinHuoApi xinHuoApi;

	private final MetadataMode metadataMode;

	/**
	 * Observation registry used for instrumentation.
	 */
	private final ObservationRegistry observationRegistry;

	/**
	 * Conventions to use for generating observations.
	 */
	private EmbeddingModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	/**
	 * Constructor for the QianFanEmbeddingModel class.
	 * @param xinHuoApi The QianFanApi instance to use for making API requests.
	 */
	public XinHuoEmbeddingModel(XinHuoApi xinHuoApi) {
		this(xinHuoApi, MetadataMode.EMBED);
	}

	/**
	 * Initializes a new instance of the QianFanEmbeddingModel class.
	 * @param xinHuoApi The QianFanApi instance to use for making API requests.
	 * @param metadataMode The mode for generating metadata.
	 */
	public XinHuoEmbeddingModel(XinHuoApi xinHuoApi, MetadataMode metadataMode) {
		this(xinHuoApi, metadataMode,
				org.springframework.ai.qianfan.XinHuoEmbeddingOptions.builder().withModel(XinHuoApi.DEFAULT_EMBEDDING_MODEL).build());
	}

	/**
	 * Initializes a new instance of the QianFanEmbeddingModel class.
	 * @param xinHuoApi The QianFanApi instance to use for making API requests.
	 * @param metadataMode The mode for generating metadata.
	 * @param xinHuoEmbeddingOptions The options for QianFan embedding.
	 */
	public XinHuoEmbeddingModel(XinHuoApi xinHuoApi, MetadataMode metadataMode,
			org.springframework.ai.qianfan.XinHuoEmbeddingOptions xinHuoEmbeddingOptions) {
		this(xinHuoApi, metadataMode, xinHuoEmbeddingOptions, RetryUtils.DEFAULT_RETRY_TEMPLATE);
	}

	/**
	 * Initializes a new instance of the QianFanEmbeddingModel class.
	 * @param xinHuoApi The QianFanApi instance to use for making API requests.
	 * @param metadataMode The mode for generating metadata.
	 * @param xinHuoEmbeddingOptions The options for QianFan embedding.
	 * @param retryTemplate - The RetryTemplate for retrying failed API requests.
	 */
	public XinHuoEmbeddingModel(XinHuoApi xinHuoApi, MetadataMode metadataMode,
			org.springframework.ai.qianfan.XinHuoEmbeddingOptions xinHuoEmbeddingOptions, RetryTemplate retryTemplate) {
		this(xinHuoApi, metadataMode, xinHuoEmbeddingOptions, retryTemplate, ObservationRegistry.NOOP);
	}

	/**
	 * Initializes a new instance of the QianFanEmbeddingModel class.
	 * @param xinHuoApi - The QianFanApi instance to use for making API requests.
	 * @param metadataMode - The mode for generating metadata.
	 * @param options - The options for QianFan embedding.
	 * @param retryTemplate - The RetryTemplate for retrying failed API requests.
	 * @param observationRegistry - The ObservationRegistry used for instrumentation.
	 */
	public XinHuoEmbeddingModel(XinHuoApi xinHuoApi, MetadataMode metadataMode, org.springframework.ai.qianfan.XinHuoEmbeddingOptions options,
			RetryTemplate retryTemplate, ObservationRegistry observationRegistry) {
		Assert.notNull(xinHuoApi, "QianFanApi must not be null");
		Assert.notNull(metadataMode, "metadataMode must not be null");
		Assert.notNull(options, "options must not be null");
		Assert.notNull(retryTemplate, "retryTemplate must not be null");
		Assert.notNull(observationRegistry, "observationRegistry must not be null");

		this.xinHuoApi = xinHuoApi;
		this.metadataMode = metadataMode;
		this.defaultOptions = options;
		this.retryTemplate = retryTemplate;
		this.observationRegistry = observationRegistry;
	}

	@Override
	public float[] embed(Document document) {
		Assert.notNull(document, "Document must not be null");
		return this.embed(document.getFormattedContent(this.metadataMode));
	}

	@Override
	public EmbeddingResponse call(EmbeddingRequest request) {
		org.springframework.ai.qianfan.XinHuoEmbeddingOptions requestOptions = mergeOptions(request.getOptions(), this.defaultOptions);
		XinHuoApi.EmbeddingRequest apiRequest = new XinHuoApi.EmbeddingRequest(request.getInstructions(),
				requestOptions.getModel(), requestOptions.getUser());

		var observationContext = EmbeddingModelObservationContext.builder()
			.embeddingRequest(request)
			.provider(XinHuoConstants.PROVIDER_NAME)
			.requestOptions(requestOptions)
			.build();

		return EmbeddingModelObservationDocumentation.EMBEDDING_MODEL_OPERATION
			.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> {
				EmbeddingList apiEmbeddingResponse = this.retryTemplate
					.execute(ctx -> this.xinHuoApi.embeddings(apiRequest).getBody());

				if (apiEmbeddingResponse == null) {
					logger.warn("No embeddings returned for request: {}", request);
					return new EmbeddingResponse(List.of());
				}

				if (apiEmbeddingResponse.errorNsg() != null) {
					logger.error("Error message returned for request: {}", apiEmbeddingResponse.errorNsg());
					throw new RuntimeException("Embedding failed: error code:" + apiEmbeddingResponse.errorCode()
							+ ", message:" + apiEmbeddingResponse.errorNsg());
				}

				var metadata = new EmbeddingResponseMetadata(apiRequest.model(),
						XinHuoUsage.from(apiEmbeddingResponse.usage()));

				List<Embedding> embeddings = apiEmbeddingResponse.data()
					.stream()
					.map(e -> new Embedding(e.embedding(), e.index()))
					.toList();

				EmbeddingResponse embeddingResponse = new EmbeddingResponse(embeddings, metadata);

				observationContext.setResponse(embeddingResponse);

				return embeddingResponse;
			});

	}

	/**
	 * Merge runtime and default {@link EmbeddingOptions} to compute the final options to
	 * use in the request.
	 */
	private org.springframework.ai.qianfan.XinHuoEmbeddingOptions mergeOptions(@Nullable EmbeddingOptions runtimeOptions,
			org.springframework.ai.qianfan.XinHuoEmbeddingOptions defaultOptions) {
		var runtimeOptionsForProvider = ModelOptionsUtils.copyToTarget(runtimeOptions, EmbeddingOptions.class,
				org.springframework.ai.qianfan.XinHuoEmbeddingOptions.class);

		if (runtimeOptionsForProvider == null) {
			return defaultOptions;
		}

		return org.springframework.ai.qianfan.XinHuoEmbeddingOptions.builder()
			.withModel(ModelOptionsUtils.mergeOption(runtimeOptionsForProvider.getModel(), defaultOptions.getModel()))
			.withUser(ModelOptionsUtils.mergeOption(runtimeOptionsForProvider.getUser(), defaultOptions.getUser()))
			.build();
	}

	public void setObservationConvention(EmbeddingModelObservationConvention observationConvention) {
		this.observationConvention = observationConvention;
	}

}

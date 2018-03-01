# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Smoke2d.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Smoke2d.proto',
  package='ssv',
  syntax='proto3',
  serialized_pb=_b('\n\rSmoke2d.proto\x12\x03ssv\"+\n\x11Smoke2dInitParams\x12\n\n\x02nx\x18\x01 \x01(\r\x12\n\n\x02ny\x18\x02 \x01(\r\"\x13\n\x11Smoke2dStepParams\"\x14\n\x12Smoke2dResetParams\"\x16\n\x14Smoke2dDestroyParams\"(\n\x14Smoke2dGetDataParams\x12\x10\n\x08property\x18\x01 \x01(\r\"\x18\n\x06Result\x12\x0e\n\x06status\x18\x01 \x01(\r\"\x19\n\tDataChunk\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x32\x87\x02\n\x07Smoke2d\x12-\n\x04Init\x12\x16.ssv.Smoke2dInitParams\x1a\x0b.ssv.Result\"\x00\x12-\n\x04Step\x12\x16.ssv.Smoke2dStepParams\x1a\x0b.ssv.Result\"\x00\x12/\n\x05Reset\x12\x17.ssv.Smoke2dResetParams\x1a\x0b.ssv.Result\"\x00\x12\x33\n\x07\x44\x65stroy\x12\x19.ssv.Smoke2dDestroyParams\x1a\x0b.ssv.Result\"\x00\x12\x38\n\x07GetData\x12\x19.ssv.Smoke2dGetDataParams\x1a\x0e.ssv.DataChunk\"\x00\x30\x01\x62\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_SMOKE2DINITPARAMS = _descriptor.Descriptor(
  name='Smoke2dInitParams',
  full_name='ssv.Smoke2dInitParams',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nx', full_name='ssv.Smoke2dInitParams.nx', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ny', full_name='ssv.Smoke2dInitParams.ny', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=65,
)


_SMOKE2DSTEPPARAMS = _descriptor.Descriptor(
  name='Smoke2dStepParams',
  full_name='ssv.Smoke2dStepParams',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=67,
  serialized_end=86,
)


_SMOKE2DRESETPARAMS = _descriptor.Descriptor(
  name='Smoke2dResetParams',
  full_name='ssv.Smoke2dResetParams',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=88,
  serialized_end=108,
)


_SMOKE2DDESTROYPARAMS = _descriptor.Descriptor(
  name='Smoke2dDestroyParams',
  full_name='ssv.Smoke2dDestroyParams',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=110,
  serialized_end=132,
)


_SMOKE2DGETDATAPARAMS = _descriptor.Descriptor(
  name='Smoke2dGetDataParams',
  full_name='ssv.Smoke2dGetDataParams',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='property', full_name='ssv.Smoke2dGetDataParams.property', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=134,
  serialized_end=174,
)


_RESULT = _descriptor.Descriptor(
  name='Result',
  full_name='ssv.Result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='ssv.Result.status', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=176,
  serialized_end=200,
)


_DATACHUNK = _descriptor.Descriptor(
  name='DataChunk',
  full_name='ssv.DataChunk',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='ssv.DataChunk.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=202,
  serialized_end=227,
)

DESCRIPTOR.message_types_by_name['Smoke2dInitParams'] = _SMOKE2DINITPARAMS
DESCRIPTOR.message_types_by_name['Smoke2dStepParams'] = _SMOKE2DSTEPPARAMS
DESCRIPTOR.message_types_by_name['Smoke2dResetParams'] = _SMOKE2DRESETPARAMS
DESCRIPTOR.message_types_by_name['Smoke2dDestroyParams'] = _SMOKE2DDESTROYPARAMS
DESCRIPTOR.message_types_by_name['Smoke2dGetDataParams'] = _SMOKE2DGETDATAPARAMS
DESCRIPTOR.message_types_by_name['Result'] = _RESULT
DESCRIPTOR.message_types_by_name['DataChunk'] = _DATACHUNK

Smoke2dInitParams = _reflection.GeneratedProtocolMessageType('Smoke2dInitParams', (_message.Message,), dict(
  DESCRIPTOR = _SMOKE2DINITPARAMS,
  __module__ = 'Smoke2d_pb2'
  # @@protoc_insertion_point(class_scope:ssv.Smoke2dInitParams)
  ))
_sym_db.RegisterMessage(Smoke2dInitParams)

Smoke2dStepParams = _reflection.GeneratedProtocolMessageType('Smoke2dStepParams', (_message.Message,), dict(
  DESCRIPTOR = _SMOKE2DSTEPPARAMS,
  __module__ = 'Smoke2d_pb2'
  # @@protoc_insertion_point(class_scope:ssv.Smoke2dStepParams)
  ))
_sym_db.RegisterMessage(Smoke2dStepParams)

Smoke2dResetParams = _reflection.GeneratedProtocolMessageType('Smoke2dResetParams', (_message.Message,), dict(
  DESCRIPTOR = _SMOKE2DRESETPARAMS,
  __module__ = 'Smoke2d_pb2'
  # @@protoc_insertion_point(class_scope:ssv.Smoke2dResetParams)
  ))
_sym_db.RegisterMessage(Smoke2dResetParams)

Smoke2dDestroyParams = _reflection.GeneratedProtocolMessageType('Smoke2dDestroyParams', (_message.Message,), dict(
  DESCRIPTOR = _SMOKE2DDESTROYPARAMS,
  __module__ = 'Smoke2d_pb2'
  # @@protoc_insertion_point(class_scope:ssv.Smoke2dDestroyParams)
  ))
_sym_db.RegisterMessage(Smoke2dDestroyParams)

Smoke2dGetDataParams = _reflection.GeneratedProtocolMessageType('Smoke2dGetDataParams', (_message.Message,), dict(
  DESCRIPTOR = _SMOKE2DGETDATAPARAMS,
  __module__ = 'Smoke2d_pb2'
  # @@protoc_insertion_point(class_scope:ssv.Smoke2dGetDataParams)
  ))
_sym_db.RegisterMessage(Smoke2dGetDataParams)

Result = _reflection.GeneratedProtocolMessageType('Result', (_message.Message,), dict(
  DESCRIPTOR = _RESULT,
  __module__ = 'Smoke2d_pb2'
  # @@protoc_insertion_point(class_scope:ssv.Result)
  ))
_sym_db.RegisterMessage(Result)

DataChunk = _reflection.GeneratedProtocolMessageType('DataChunk', (_message.Message,), dict(
  DESCRIPTOR = _DATACHUNK,
  __module__ = 'Smoke2d_pb2'
  # @@protoc_insertion_point(class_scope:ssv.DataChunk)
  ))
_sym_db.RegisterMessage(DataChunk)


try:
  # THESE ELEMENTS WILL BE DEPRECATED.
  # Please use the generated *_pb2_grpc.py files instead.
  import grpc
  from grpc.beta import implementations as beta_implementations
  from grpc.beta import interfaces as beta_interfaces
  from grpc.framework.common import cardinality
  from grpc.framework.interfaces.face import utilities as face_utilities


  class Smoke2dStub(object):

    def __init__(self, channel):
      """Constructor.

      Args:
        channel: A grpc.Channel.
      """
      self.Init = channel.unary_unary(
          '/ssv.Smoke2d/Init',
          request_serializer=Smoke2dInitParams.SerializeToString,
          response_deserializer=Result.FromString,
          )
      self.Step = channel.unary_unary(
          '/ssv.Smoke2d/Step',
          request_serializer=Smoke2dStepParams.SerializeToString,
          response_deserializer=Result.FromString,
          )
      self.Reset = channel.unary_unary(
          '/ssv.Smoke2d/Reset',
          request_serializer=Smoke2dResetParams.SerializeToString,
          response_deserializer=Result.FromString,
          )
      self.Destroy = channel.unary_unary(
          '/ssv.Smoke2d/Destroy',
          request_serializer=Smoke2dDestroyParams.SerializeToString,
          response_deserializer=Result.FromString,
          )
      self.GetData = channel.unary_stream(
          '/ssv.Smoke2d/GetData',
          request_serializer=Smoke2dGetDataParams.SerializeToString,
          response_deserializer=DataChunk.FromString,
          )


  class Smoke2dServicer(object):

    def Init(self, request, context):
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')

    def Step(self, request, context):
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')

    def Reset(self, request, context):
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')

    def Destroy(self, request, context):
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')

    def GetData(self, request, context):
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')


  def add_Smoke2dServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Init': grpc.unary_unary_rpc_method_handler(
            servicer.Init,
            request_deserializer=Smoke2dInitParams.FromString,
            response_serializer=Result.SerializeToString,
        ),
        'Step': grpc.unary_unary_rpc_method_handler(
            servicer.Step,
            request_deserializer=Smoke2dStepParams.FromString,
            response_serializer=Result.SerializeToString,
        ),
        'Reset': grpc.unary_unary_rpc_method_handler(
            servicer.Reset,
            request_deserializer=Smoke2dResetParams.FromString,
            response_serializer=Result.SerializeToString,
        ),
        'Destroy': grpc.unary_unary_rpc_method_handler(
            servicer.Destroy,
            request_deserializer=Smoke2dDestroyParams.FromString,
            response_serializer=Result.SerializeToString,
        ),
        'GetData': grpc.unary_stream_rpc_method_handler(
            servicer.GetData,
            request_deserializer=Smoke2dGetDataParams.FromString,
            response_serializer=DataChunk.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'ssv.Smoke2d', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


  class BetaSmoke2dServicer(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    def Init(self, request, context):
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
    def Step(self, request, context):
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
    def Reset(self, request, context):
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
    def Destroy(self, request, context):
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
    def GetData(self, request, context):
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)


  class BetaSmoke2dStub(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    def Init(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      raise NotImplementedError()
    Init.future = None
    def Step(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      raise NotImplementedError()
    Step.future = None
    def Reset(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      raise NotImplementedError()
    Reset.future = None
    def Destroy(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      raise NotImplementedError()
    Destroy.future = None
    def GetData(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      raise NotImplementedError()


  def beta_create_Smoke2d_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_deserializers = {
      ('ssv.Smoke2d', 'Destroy'): Smoke2dDestroyParams.FromString,
      ('ssv.Smoke2d', 'GetData'): Smoke2dGetDataParams.FromString,
      ('ssv.Smoke2d', 'Init'): Smoke2dInitParams.FromString,
      ('ssv.Smoke2d', 'Reset'): Smoke2dResetParams.FromString,
      ('ssv.Smoke2d', 'Step'): Smoke2dStepParams.FromString,
    }
    response_serializers = {
      ('ssv.Smoke2d', 'Destroy'): Result.SerializeToString,
      ('ssv.Smoke2d', 'GetData'): DataChunk.SerializeToString,
      ('ssv.Smoke2d', 'Init'): Result.SerializeToString,
      ('ssv.Smoke2d', 'Reset'): Result.SerializeToString,
      ('ssv.Smoke2d', 'Step'): Result.SerializeToString,
    }
    method_implementations = {
      ('ssv.Smoke2d', 'Destroy'): face_utilities.unary_unary_inline(servicer.Destroy),
      ('ssv.Smoke2d', 'GetData'): face_utilities.unary_stream_inline(servicer.GetData),
      ('ssv.Smoke2d', 'Init'): face_utilities.unary_unary_inline(servicer.Init),
      ('ssv.Smoke2d', 'Reset'): face_utilities.unary_unary_inline(servicer.Reset),
      ('ssv.Smoke2d', 'Step'): face_utilities.unary_unary_inline(servicer.Step),
    }
    server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
    return beta_implementations.server(method_implementations, options=server_options)


  def beta_create_Smoke2d_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_serializers = {
      ('ssv.Smoke2d', 'Destroy'): Smoke2dDestroyParams.SerializeToString,
      ('ssv.Smoke2d', 'GetData'): Smoke2dGetDataParams.SerializeToString,
      ('ssv.Smoke2d', 'Init'): Smoke2dInitParams.SerializeToString,
      ('ssv.Smoke2d', 'Reset'): Smoke2dResetParams.SerializeToString,
      ('ssv.Smoke2d', 'Step'): Smoke2dStepParams.SerializeToString,
    }
    response_deserializers = {
      ('ssv.Smoke2d', 'Destroy'): Result.FromString,
      ('ssv.Smoke2d', 'GetData'): DataChunk.FromString,
      ('ssv.Smoke2d', 'Init'): Result.FromString,
      ('ssv.Smoke2d', 'Reset'): Result.FromString,
      ('ssv.Smoke2d', 'Step'): Result.FromString,
    }
    cardinalities = {
      'Destroy': cardinality.Cardinality.UNARY_UNARY,
      'GetData': cardinality.Cardinality.UNARY_STREAM,
      'Init': cardinality.Cardinality.UNARY_UNARY,
      'Reset': cardinality.Cardinality.UNARY_UNARY,
      'Step': cardinality.Cardinality.UNARY_UNARY,
    }
    stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
    return beta_implementations.dynamic_stub(channel, 'ssv.Smoke2d', cardinalities, options=stub_options)
except ImportError:
  pass
# @@protoc_insertion_point(module_scope)

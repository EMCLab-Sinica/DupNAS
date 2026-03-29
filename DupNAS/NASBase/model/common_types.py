import enum


#######################
# Enums
#######################

class Mat:
   def __init__(self, data, n, ch, h, w):
        self.data = data
        self.n = n
        self.ch = ch
        self.h = h
        self.w = w

   def __repr__(self):
        s = "MAT(data = " + str(self.data) + \
              ", n = " + str(self.n) + \
              ", ch = " + str(self.ch) + \
              ", h = " + str(self.h) + \
              ", w = " + str(self.w) + ")\n"
        return s
     
   def __str__(self):
        s = "MAT(data = " + str(self.data) + \
              ", n = " + str(self.n) + \
              ", ch = " + str(self.ch) + \
              ", h = " + str(self.h) + \
              ", w = " + str(self.w) + ")\n"
        return s
     


class LAYERTYPES(int, enum.Enum):
   
   # conv types
   L_CONV     = 0
   #L_SEPCONV  = 1
      
   
   L_MBCONV = 1
   L_MBCONV_1D = 2

   L_SHUFFLE = 3

   L_INCEPTA = 4
   L_INCEPTB = 5
   L_INCEPTC = 6
   # L_MBCONV1  = 1
   # L_MBCONV2  = 2
   # L_MBCONV3  = 3
   # L_MBCONV4  = 4
   # L_MBCONV5  = 5
   # L_MBCONV6  = 6
   # L_MBCONV7  = 7
   # L_MBCONV8  = 8
   
   # others
   L_BN       = 10
   L_RELU     = 20
   L_FC       = 30
   L_AVGPOOL  = 40
   L_ADD      = 50
   
   
class OPTYPES(int, enum.Enum):
    # different conv operations
   O_CONV2D      = 0
   O_CONV2D_SP   = 1   # seperable
   O_CONV2D_PW   = 2   # pointwise
   O_CONV2D_DW   = 3   # depthwise
   
   O_BN          = 4
   O_RELU        = 5
   O_FC          = 6   
   O_AVGPOOL     = 7
   
   O_ADD        = 8 # residual aggregation (summation)

   O_CONV1D      = 9
   O_CONV1D_SP   = 10
   O_CONV1D_PW   = 11
   O_CONV1D_DW   = 12

   
   
   @staticmethod
   def get_conv_optype_by_name(mod_name):
      if ("_pw" in mod_name):
         return OPTYPES.O_CONV2D_PW # pointwise
      elif ("_dw" in mod_name):
         return OPTYPES.O_CONV2D_DW # depthwise
      elif ("_sp" in mod_name):
         return OPTYPES.O_CONV2D_SP # seperable
      else:
         return OPTYPES.O_CONV2D    # standard
   
   def get_conv1d_optype_by_name(mod_name):
      if ("_pw" in mod_name):
         return OPTYPES.O_CONV1D_PW # pointwise
      elif ("_dw" in mod_name):
         return OPTYPES.O_CONV1D_DW # depthwise
      elif ("_sp" in mod_name):
         return OPTYPES.O_CONV1D_SP # seperable
      else:
         return OPTYPES.O_CONV1D    # standard

   @staticmethod
   def get_optype_label(op_type):
      if op_type == OPTYPES.O_CONV2D:
         return "O_CONV2D"
      elif op_type == OPTYPES.O_CONV2D_SP:
         return "O_CONV2D_SP"
      elif op_type == OPTYPES.O_CONV2D_PW:
         return "O_CONV2D_PW"
      elif op_type == OPTYPES.O_CONV2D_DW:
         return "O_CONV2D_DW"
      elif op_type == OPTYPES.O_CONV1D:
         return "O_CONV1D"
      elif op_type == OPTYPES.O_CONV1D_SP:
         return "O_CONV1D_SP"
      elif op_type == OPTYPES.O_CONV1D_PW:
         return "O_CONV1D_PW"
      elif op_type == OPTYPES.O_CONV1D_DW:
         return "O_CONV1D_DW"
      elif op_type == OPTYPES.O_BN:
         return "O_BN"
      elif op_type == OPTYPES.O_RELU:
         return "O_RELU"
      elif op_type == OPTYPES.O_FC:
         return "O_FC"
      elif op_type == OPTYPES.O_AVGPOOL:
         return "O_AVGPOOL"         
      elif op_type == OPTYPES.O_ADD:
         return "O_ADD"            
      else:
         return "UNKNOWN_OP_TYPE"
      

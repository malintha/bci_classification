% data seperate by classes
function[data] = seperateClasses(x,y)
   [m,n,o]=size(x);
   one =[];
   two =[];
   three = [];
   four = [];
   for i=1:o
       if y(i,:)==1
           one= cat(3,one ,x(:,:,i));
       elseif y(i,:)==2
           two= cat(3,two ,x(:,:,i));
       elseif y(i,:)==3
           three= cat(3,three ,x(:,:,i));
       else
           four= cat(3,four ,x(:,:,i));
       end
   end
   data.one=one;
   data.two=two;
   data.three=three;
   data.four=four;
end